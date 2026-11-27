"""Byte-pair encoding (BPE) tokenizer, learned from data/train.txt at the start of train.py.

Tokens 0-255 are the raw bytes, so any byte string can be encoded and decoding is exact. Token 256 + k is
merge k: the concatenation of an earlier pair of tokens (a, b). The last id is the beginning-of-sequence token.
The only file is out/merges.npy, a (n_merges, 2) int32 array (a few KB towards the size limit).

Training and encoding both work on the whole token stream at once on the GPU: one merge is a handful of
vectorised passes (find every (a, b), replace it with the new token, drop the second half of each pair).
Applying the merges in order like this gives exactly the standard BPE encoding.
"""
import os

import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK = 1 << 26  # bytes encoded at once (bounds GPU memory; a pair cut at a chunk edge just stays unmerged)


def _merge(x, a, b, new):
    """Replace each occurrence of the pair (a, b) in the 1D tensor x by the token new, left to right."""
    m = (x[:-1] == a) & (x[1:] == b)
    if a == b:  # in a run like "a a a" the matches overlap: merge the 1st, 3rd, ... of each run
        c = m.cumsum(0)
        before_run = torch.where(m, 0, c).cummax(0).values  # matches before the current run
        m &= (c - before_run) % 2 == 1
    i = m.nonzero().squeeze(1)
    if i.numel() == 0:
        return x
    x[i] = new
    keep = torch.ones(len(x), dtype=torch.bool, device=x.device)
    keep[i + 1] = False
    return x[keep]


def _to_tensor(text):
    return torch.from_numpy(np.frombuffer(text, dtype=np.uint8).astype(np.int32)).to(DEVICE)


class BPETokenizer:
    def __init__(self, merges):
        self.merges = np.asarray(merges, dtype=np.int32).reshape(-1, 2)
        self.vocab_size = 256 + len(self.merges) + 1
        self.bos = self.vocab_size - 1
        self.table = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self.table.append(self.table[a] + self.table[b])
        self.table.append(b"")  # BOS decodes to nothing

    def encode(self, text: bytes) -> np.ndarray:
        out = []
        for s in range(0, len(text), CHUNK):
            x = _to_tensor(text[s : s + CHUNK])
            for k, (a, b) in enumerate(self.merges.tolist()):
                x = _merge(x, a, b, 256 + k)
            out.append(x.to(torch.int16 if self.vocab_size <= 32768 else torch.int32).cpu().numpy())
        ids = np.concatenate(out) if out else np.zeros(0, np.int16)
        return ids.astype(np.uint16 if self.vocab_size <= 65536 else np.int64)

    def decode(self, ids) -> bytes:
        table = self.table
        return b"".join([table[i] for i in np.asarray(ids).reshape(-1).tolist()])

    def save(self, out_dir):
        np.save(os.path.join(out_dir, "merges.npy"), self.merges)


def train_tokenizer(text: bytes, vocab_size: int) -> BPETokenizer:
    """Greedy BPE: repeatedly merge the most frequent adjacent pair of tokens in text."""
    x = _to_tensor(text)
    merges = []
    for new in range(256, vocab_size - 1):  # the last id is reserved for BOS
        pair = int(torch.bincount(x[:-1] * vocab_size + x[1:]).argmax())
        a, b = divmod(pair, vocab_size)
        merges.append((a, b))
        x = _merge(x, a, b, new)
    return BPETokenizer(merges)


def load_tokenizer(out_dir):
    return BPETokenizer(np.load(os.path.join(out_dir, "merges.npy")))
