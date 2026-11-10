"""Vectorized feedback table: pattern code for every (guess, answer) pair.

12,972 guesses x 2,315 answers is about 30 MB as uint8 and takes a few seconds to
build. The table is cached on disk, keyed by a checksum of both word lists, so a
changed list can never reuse a stale table.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np

_POW = np.array([81, 27, 9, 3, 1], dtype=np.uint16)


def words_to_array(words) -> np.ndarray:
    buf = "".join(words).encode("ascii")
    return (np.frombuffer(buf, dtype=np.uint8).reshape(-1, 5) - ord("a")).astype(np.int8)


def pattern_matrix(guesses, answers, chunk: int = 256) -> np.ndarray:
    G = words_to_array(guesses)
    A = words_to_array(answers)
    out = np.empty((len(G), len(A)), dtype=np.uint8)
    for s in range(0, len(G), chunk):
        g = G[s : s + chunk]
        c = len(g)
        green = g[:, None, :] == A[None, :, :]  # (c, n, 5)
        not_green = ~green
        # avail[i]: copies of letter g_i in the answer at non-green positions
        avail = np.zeros((c, len(A), 5), dtype=np.int8)
        for k in range(5):
            ak_free = not_green[:, :, k]
            for i in range(5):
                avail[:, :, i] += (g[:, None, i] == A[None, :, k]) & ak_free
        # earlier[i]: non-green positions j < i in the guess with the same letter
        earlier = np.zeros_like(avail)
        for i in range(5):
            for j in range(i):
                same = (g[:, j] == g[:, i])[:, None]
                earlier[:, :, i] += same & not_green[:, :, j]
        yellow = not_green & (avail > earlier)
        vals = green.astype(np.uint16) * 2 + yellow.astype(np.uint16)
        out[s : s + c] = (vals * _POW).sum(-1).astype(np.uint8)
    return out


def cache_dir() -> Path:
    root = os.environ.get("FL_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "fiveletters")
    p = Path(root)
    p.mkdir(parents=True, exist_ok=True)
    return p


def cached_pattern_matrix(guesses, answers) -> np.ndarray:
    key = hashlib.sha256(("\n".join(guesses) + "|" + "\n".join(answers)).encode()).hexdigest()[:16]
    path = cache_dir() / f"patterns-{len(guesses)}x{len(answers)}-{key}.npy"
    if path.exists():
        try:
            m = np.load(path)
            if m.shape == (len(guesses), len(answers)) and m.dtype == np.uint8:
                return m
        except (OSError, ValueError):
            pass  # truncated or corrupt cache file: rebuild below
    m = pattern_matrix(guesses, answers)
    # Atomic write: a crash mid-save never leaves a half-written cache behind.
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".npy.tmp")
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            np.save(f, m)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return m
