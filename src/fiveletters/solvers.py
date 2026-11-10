"""Teacher and baseline solvers.

``EntropySolver`` picks the guess whose feedback splits the remaining candidates
most evenly (maximum entropy). With a small bonus for guesses that could be the
answer, it plays close to the known optimum of 3.421 mean guesses. It is
deterministic, so its decisions are memoized by history; this makes trace
generation and full evaluations fast.

``RandomConsistentSolver`` guesses a uniformly random word that fits every clue.
It is the "just track the constraints" baseline.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
import tempfile
from pathlib import Path

import numpy as np

from .feedback import N_PATTERNS, encode
from .game import Turn
from .patterns import cache_dir, cached_pattern_matrix
from .words import WordLists, guess_vocab, load, sha256_words


class _Base:
    def __init__(self, pool: str = "answers", hard_mode: bool = False, wl: WordLists | None = None,
                 vocab: str = "wordle"):
        if pool not in ("answers", "all"):
            raise ValueError("pool must be 'answers' or 'all'")
        self.wl = wl or load()
        self.pool_name = pool
        self.hard_mode = hard_mode
        self.guesses = list(self.wl.guesses)
        # Answers come first in `guesses`, so pool index i is also guess index i.
        self.pool = list(self.wl.answers) if pool == "answers" else list(self.wl.guesses)
        self.gidx = {w: i for i, w in enumerate(self.guesses)}
        self.M = cached_pattern_matrix(self.guesses, self.pool)
        self.vocab = vocab
        allowed = set(guess_vocab(vocab))
        self.allowed = np.array([i for i, w in enumerate(self.guesses) if w in allowed])

    def candidates(self, turns: list[Turn]) -> np.ndarray:
        mask = np.ones(len(self.pool), dtype=bool)
        for t in turns:
            if t.valid:
                mask &= self.M[self.gidx[t.guess]] == encode(t.feedback)
        cands = np.flatnonzero(mask)
        if len(cands) == 0:
            raise ValueError(
                f"no {self.pool_name!r} word fits these clues; use pool='all' for answers outside the answer list"
            )
        return cands

    def describe(self) -> dict:
        return {"solver": type(self).__name__, "pool": self.pool_name, "hard_mode": self.hard_mode, "vocab": self.vocab}


class EntropySolver(_Base):
    def __init__(self, pool: str = "answers", hard_mode: bool = False, candidate_bonus: float = 0.1,
                 opener: str | None = None, wl: WordLists | None = None, vocab: str = "wordle"):
        super().__init__(pool, hard_mode, wl, vocab)
        self.candidate_bonus = candidate_bonus
        self._cache: dict[tuple, str] = {}
        self._opener = opener

    def describe(self) -> dict:
        return {**super().describe(), "candidate_bonus": self.candidate_bonus, "opener": self.opener}

    @property
    def opener(self) -> str:
        if self._opener is None:
            self._opener = self.ranked_openers(1)[0]
        return self._opener

    def _scores(self, guess_ids: np.ndarray, cands: np.ndarray) -> np.ndarray:
        sub = self.M[np.ix_(guess_ids, cands)].astype(np.int64)
        sub += (np.arange(len(guess_ids), dtype=np.int64) * N_PATTERNS)[:, None]
        counts = np.bincount(sub.ravel(), minlength=len(guess_ids) * N_PATTERNS).reshape(len(guess_ids), N_PATTERNS)
        p = counts / len(cands)
        with np.errstate(divide="ignore", invalid="ignore"):
            h = -np.where(p > 0, p * np.log2(p), 0.0).sum(1)
        is_cand = np.isin(guess_ids, cands)
        return h + self.candidate_bonus * is_cand

    def ranked_openers(self, k: int = 50) -> list[str]:
        cands = np.arange(len(self.pool))
        gids = cands if self.hard_mode else self.allowed
        scores = self._scores(gids, cands)
        order = np.argsort(-scores, kind="stable")[:k]
        return [self.guesses[gids[i]] for i in order]

    def choose(self, turns: list[Turn]) -> str:
        key = tuple((t.guess, encode(t.feedback)) for t in turns if t.valid)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        if not key:
            word = self.opener
        else:
            cands = self.candidates(turns)
            if len(cands) <= 2:
                word = self.pool[cands[0]]
            else:
                # Hard mode: candidates satisfy every clue, so they are always legal guesses.
                gids = cands if self.hard_mode else self.allowed
                scores = self._scores(gids, cands)
                word = self.guesses[gids[int(np.argmax(scores))]]
        self._cache[key] = word
        return word


class LookaheadSolver(EntropySolver):
    """Beam search over the game tree, scored by total guesses summed over all candidates.

    At each state it tries the ``beam[depth]`` best guesses by entropy score and recurses
    into every feedback branch, keeping the guess with the lowest expected guess count.
    Results are memoized by (candidate set, turns used). Histories it has never seen,
    such as random curriculum prefixes, are handled the same way, because the search
    depends only on the remaining candidates.
    """

    INF = 10**9

    def __init__(self, pool: str = "answers", hard_mode: bool = False, beam=(1, 12, 6, 3, 2, 1),
                 opener: str | None = "salet", max_turns: int = 6, wl: WordLists | None = None, vocab: str = "wordle"):
        if vocab != "wordle" and opener == "salet":
            opener = None  # SALET is outside the shared vocabulary: pick the best opener inside it
        super().__init__(pool=pool, hard_mode=hard_mode, candidate_bonus=0.1, opener=opener, wl=wl, vocab=vocab)
        self.beam = tuple(beam)
        self.max_turns = max_turns
        self._memo: dict[tuple, tuple[int, int]] = {}
        self._memo_loaded = 0
        self._load_memo()

    def describe(self) -> dict:
        return {**super().describe(), "beam": list(self.beam)}

    def _memo_path(self) -> Path:
        ident = json.dumps([self.pool_name, self.hard_mode, self.beam, self.opener, self.max_turns, self.vocab,
                            sha256_words(self.guesses)])
        return cache_dir() / f"lookahead-{hashlib.sha256(ident.encode()).hexdigest()[:16]}.pkl"

    def _load_memo(self) -> None:
        path = self._memo_path()
        if not path.exists():
            return
        try:
            with open(path, "rb") as f:
                self._memo.update(pickle.load(f))  # our own cache file, written by save_memo
            self._memo_loaded = len(self._memo)
        except (OSError, EOFError, pickle.UnpicklingError):
            pass  # corrupt cache: recompute

    def save_memo(self) -> None:
        """Persist search results so the next session (or worker) skips the slow search. Atomic."""
        if len(self._memo) <= self._memo_loaded:
            return
        path = self._memo_path()
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".pkl.tmp")
        with os.fdopen(fd, "wb") as f:
            pickle.dump(self._memo, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
        self._memo_loaded = len(self._memo)

    def _best(self, cands: tuple[int, ...], used: int) -> tuple[int, int]:
        """(total guesses over all candidates, guess id) when `used` turns are already spent."""
        n = len(cands)
        left = self.max_turns - used
        if left <= 0:
            return self.INF, -1
        if n == 1:
            return 1, cands[0]
        if n == 2:
            return (3, cands[0]) if left >= 2 else (self.INF, cands[0])
        key = (cands, used)
        hit = self._memo.get(key)
        if hit is not None:
            return hit
        c_arr = np.array(cands)
        gids = c_arr if self.hard_mode else self.allowed
        scores = self._scores(gids, c_arr)
        k = self.beam[min(used, len(self.beam) - 1)]
        order = np.argsort(-scores, kind="stable")[: max(k, 1)]
        best = (self.INF, int(gids[order[0]]))
        for oi in order:
            g = int(gids[oi])
            pats = self.M[g, c_arr]
            if len(np.unique(pats)) == 1 and g not in cands:
                continue  # learns nothing
            total = n
            for p in np.unique(pats):
                if p == 242:
                    continue
                sub = tuple(int(x) for x in c_arr[pats == p])
                sub_cost, _ = self._best(sub, used + 1)
                total += sub_cost
                if total >= best[0]:
                    break
            if total < best[0]:
                best = (total, g)
        self._memo[key] = best
        return best

    def choose(self, turns: list[Turn]) -> str:
        if not turns:
            return self.opener
        cands = tuple(int(x) for x in self.candidates(turns))
        cost, g = self._best(cands, len(turns))
        if g < 0 or cost >= self.INF:
            return EntropySolver.choose(self, turns)  # out of turns for a full search: be greedy
        return self.guesses[g]


class RandomConsistentSolver(_Base):
    def __init__(self, pool: str = "answers", hard_mode: bool = False, seed: int = 0, wl: WordLists | None = None):
        super().__init__(pool, hard_mode, wl)
        self.rng = random.Random(seed)

    def choose(self, turns: list[Turn]) -> str:
        cands = self.candidates(turns)
        return self.pool[cands[self.rng.randrange(len(cands))]]


def word_zipf(words) -> np.ndarray:
    """General-English frequency (Zipf scale, 0 = unseen, 7 = 'the') from the public `wordfreq` package.
    Never derived from Wordle answer lists."""
    from wordfreq import zipf_frequency

    return np.array([zipf_frequency(w, "en") for w in words], dtype=np.float32)


class CommonProbeSolver(EntropySolver):
    """Round v6 teacher. Like the consistent teacher, every guess fits all clues, with two extra rules a small model
    can learn:

    * common words: among clue-fitting guesses of (nearly) equal information, play the most common English word, and
      when 1-2 candidates remain play the most common one (answers are everyday words; the student picked AXONE over
      ATONE because nothing taught it this);
    * probe on traps: when 3..max_probe_set candidates remain and some common word (not necessarily fitting) splits
      them into at least `probe_margin` more groups than the best fitting word does, play that word instead
      (BATCH/HATCH/LATCH/MATCH/WATCH: one probe with B, H, L, M, W letters beats guessing them one by one).
    """

    def __init__(self, pool: str = "answers", vocab: str = "shared", opener: str | None = "crate", common: bool = True,
                 probe: bool = True, zipf_min: float = 2.0, probe_margin: int = 2, max_probe_set: int = 20,
                 info_tolerance: float = 0.1, wl: WordLists | None = None):
        super().__init__(pool=pool, hard_mode=True, candidate_bonus=0.0, opener=opener, wl=wl, vocab=vocab)
        self.common, self.probe = common, probe
        self.zipf_min, self.probe_margin, self.max_probe_set = zipf_min, probe_margin, max_probe_set
        self.info_tolerance = info_tolerance
        self.zipf = word_zipf(self.guesses)
        self.probe_ids = self.allowed[self.zipf[self.allowed] >= zipf_min]

    def describe(self) -> dict:
        return {**super().describe(), "solver": "CommonProbeSolver", "common": self.common, "probe": self.probe,
                "zipf_min": self.zipf_min, "probe_margin": self.probe_margin, "max_probe_set": self.max_probe_set}

    def _groups(self, gids: np.ndarray, cands: np.ndarray) -> np.ndarray:
        sub = self.M[np.ix_(gids, cands)]
        return np.array([len(np.unique(row)) for row in sub])

    def choose(self, turns: list[Turn]) -> str:
        key = tuple((t.guess, encode(t.feedback)) for t in turns if t.valid)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        if not key:
            word = self.opener
        else:
            cands = self.candidates(turns)  # pool index == guess index (answers come first in `guesses`)
            if len(cands) <= 2:
                order = np.argsort(-self.zipf[cands], kind="stable") if self.common else np.arange(len(cands))
                word = self.pool[cands[order[0]]]
            else:
                scores = self._scores(cands, cands)
                if self.common:
                    near = np.flatnonzero(scores >= scores.max() - self.info_tolerance)
                    best = near[np.argmax(self.zipf[cands[near]])]
                else:
                    best = int(np.argmax(scores))
                word = self.pool[cands[best]]
                if self.probe and 3 <= len(cands) <= self.max_probe_set and len(self.probe_ids):
                    inside = int(self._groups(cands, cands).max())
                    groups = self._groups(self.probe_ids, cands)
                    j = int(np.argmax(groups))
                    if groups[j] >= inside + self.probe_margin:
                        word = self.guesses[self.probe_ids[j]]
        self._cache[key] = word
        return word


FAST_BEAM = (1, 3, 2, 1, 1, 1)


def make_solver(name: str, pool: str = "answers", hard_mode: bool = False, seed: int = 0, vocab: str = "wordle",
                opener: str | None = None):
    """entropy: greedy, about 3.47 mean. lookahead: beam search, 3.4216 on the 2,315 answers
    (optimal is 3.4212), slow the first time and cached after. fast-lookahead: small beam, for
    the off-tree states that curriculum prefixes and noisy traces create."""
    if name == "entropy":
        return EntropySolver(pool=pool, hard_mode=hard_mode, vocab=vocab)
    if name == "lookahead":
        return LookaheadSolver(pool=pool, hard_mode=hard_mode, vocab=vocab, **({"opener": opener} if opener else {}))
    if name == "fast-lookahead":
        return LookaheadSolver(pool=pool, hard_mode=hard_mode, beam=FAST_BEAM, vocab=vocab)
    if name == "random":
        return RandomConsistentSolver(pool=pool, hard_mode=hard_mode, seed=seed)
    if name in ("common-probe", "common-only", "probe-only", "consistent-greedy"):
        return CommonProbeSolver(pool=pool, vocab=vocab, opener=opener or "crate",
                                 common=name in ("common-probe", "common-only"),
                                 probe=name in ("common-probe", "probe-only"))
    raise ValueError(f"unknown solver {name!r}")
