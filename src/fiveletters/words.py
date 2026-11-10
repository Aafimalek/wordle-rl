"""Word lists and the frozen evaluation splits.

The answer list (2,315 words) and the extra allowed guesses (10,657 words) are the
original Wordle lists. Together they make the 12,972 valid guesses. Answers come
first in ``WordLists.guesses`` so an answer's index is the same in both lists.

Splits are generated once by ``make_splits`` and committed as text files. At load
time we check them against ``manifest.json``, so a silently edited word list or a
line-ending change can never alter the benchmark.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from functools import cache
from importlib import resources

SPLIT_SEED = 20261108
N_HELDOUT = 463  # 20% of 2,315
N_VAL = 100  # carved from the training answers, used for model selection
N_RARE = 500  # valid guesses that are never answers

SPLIT_NAMES = ("train", "val", "heldout", "full", "rare")
# Public leaderboard word sets, frozen as published (sources in manifest.json). Words of these sets may
# fall in our train split; benchmark-clean runs exclude them from training (see benchmark_words()).
BENCHMARK_NAMES = ("wordlebench", "jeffers")
# Larger model-selection set carved from train (round v5+). 100-game val proved too noisy (CI about ±9 points).
SELECTION_NAMES = ("val300", "select")
SOURCES = {
    "answers.txt": "https://gist.github.com/cfreshman/a03ef2cba789d8cf00c08f767e0fad7b",
    "extra_guesses.txt": "https://gist.github.com/cfreshman/cdcdf777450c5b5301e439061d29694c",
}
_WORD = re.compile(r"^[a-z]{5}$")


class ResourceError(RuntimeError):
    """A word list or split does not match the committed manifest."""


def _res():
    return resources.files("fiveletters.resources")


def _read_words(rel: str) -> list[str]:
    text = _res().joinpath(rel).read_text("utf-8")
    words = [w.strip() for w in text.splitlines() if w.strip()]
    bad = [w for w in words if not _WORD.match(w)]
    if bad:
        raise ResourceError(f"{rel}: {len(bad)} malformed entries, first: {bad[:3]}")
    if len(set(words)) != len(words):
        raise ResourceError(f"{rel}: duplicate words")
    return words


def sha256_words(words: list[str] | tuple[str, ...]) -> str:
    """Checksum of a word list that ignores line endings and trailing whitespace."""
    return hashlib.sha256("\n".join(words).encode("ascii")).hexdigest()


@dataclass(frozen=True)
class WordLists:
    answers: tuple[str, ...]
    extra: tuple[str, ...]
    guesses: tuple[str, ...] = field(init=False)
    guess_set: frozenset[str] = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "guesses", self.answers + self.extra)
        object.__setattr__(self, "guess_set", frozenset(self.guesses))


@cache
def load() -> WordLists:
    answers = _read_words("answers.txt")
    extra = _read_words("extra_guesses.txt")
    if set(answers) & set(extra):
        raise ResourceError("answers and extra guesses overlap")
    return WordLists(tuple(answers), tuple(extra))


VOCABS = ("wordle", "shared")


@cache
def textarena_accepted() -> frozenset[str]:
    """Wordle guesses that TextArena 0.7.4's English dictionary also accepts (8,019 words)."""
    words = _read_words("textarena_accepts.txt")
    want = manifest()["lists"]["textarena_accepts.txt"]
    if len(words) != want["count"] or sha256_words(words) != want["sha256"]:
        raise ResourceError("textarena_accepts.txt does not match manifest.json")
    return frozenset(words)


def guess_vocab(name: str = "wordle") -> list[str]:
    """Words a teacher may guess. 'wordle': all 12,972. 'shared': accepted by both Wordle and TextArena,
    plus every answer (a solver must be able to name the answer). Order follows WordLists.guesses."""
    wl = load()
    if name == "wordle":
        return list(wl.guesses)
    if name == "shared":
        ok = textarena_accepted() | set(wl.answers)
        return [w for w in wl.guesses if w in ok]
    raise KeyError(f"unknown vocab {name!r}; choose from {VOCABS}")


def extra_answer_candidates() -> list[str]:
    """Shared-vocabulary words that are not Wordle answers (so never in val/heldout/full) and not in rare."""
    wl = load()
    answers, rare = set(wl.answers), set(get_split("rare"))
    return [w for w in guess_vocab("shared") if w not in answers and w not in rare]


@cache
def _extra_answers(name: str = "extra_answers.txt") -> tuple[str, ...]:
    words = _read_words(name)
    want = manifest()["lists"][name]
    if len(words) != want["count"] or sha256_words(words) != want["sha256"]:
        raise ResourceError(f"{name} does not match manifest.json")
    if not set(words) <= set(extra_answer_candidates()):
        raise ResourceError(f"{name} contains a test or answer word")
    return tuple(words)


def extra_answer_words(broad: bool = False) -> list[str]:
    """Extra *training answers* so the model learns to find any real word that fits the clues instead of
    recalling the 1,752 training answers. No test words. Default: 1,610 words as frequent as real answers
    (zipf >= 2.5). broad=True: 2,484 words (zipf >= 2.0), a superset, to teach the spelling of more words."""
    return list(_extra_answers("extra_answers_broad.txt" if broad else "extra_answers.txt"))


def make_splits(wl: WordLists | None = None) -> dict[str, list[str]]:
    """Deterministically build every split. Only used to (re)create the committed files."""
    wl = wl or load()
    shuffled = sorted(wl.answers)
    random.Random(SPLIT_SEED).shuffle(shuffled)
    heldout = sorted(shuffled[:N_HELDOUT])
    val = sorted(shuffled[N_HELDOUT : N_HELDOUT + N_VAL])
    train = sorted(shuffled[N_HELDOUT + N_VAL :])
    rare = sorted(random.Random(SPLIT_SEED + 1).sample(sorted(wl.extra), N_RARE))
    return {"train": train, "val": val, "heldout": heldout, "full": sorted(wl.answers), "rare": rare}


def build_manifest(splits: dict[str, list[str]], wl: WordLists | None = None) -> dict:
    wl = wl or load()
    return {
        "split_seed": SPLIT_SEED,
        "lists": {
            "answers.txt": {"count": len(wl.answers), "sha256": sha256_words(wl.answers), "source": SOURCES["answers.txt"]},
            "extra_guesses.txt": {"count": len(wl.extra), "sha256": sha256_words(wl.extra), "source": SOURCES["extra_guesses.txt"]},
        },
        "splits": {name: {"count": len(ws), "sha256": sha256_words(ws)} for name, ws in splits.items()},
    }


@cache
def manifest() -> dict:
    return json.loads(_res().joinpath("manifest.json").read_text("utf-8"))


def check_resources() -> dict:
    """Verify word lists and splits against the manifest. Raises ResourceError on any mismatch."""
    wl = load()
    man = manifest()
    for name, words in (("answers.txt", wl.answers), ("extra_guesses.txt", wl.extra)):
        want = man["lists"][name]
        if len(words) != want["count"] or sha256_words(words) != want["sha256"]:
            raise ResourceError(f"{name} does not match manifest.json")
    for name in SPLIT_NAMES + BENCHMARK_NAMES + SELECTION_NAMES:
        get_split(name)
    if not set(get_split("val300")) <= set(get_split("train")) or set(get_split("val300")) & benchmark_words():
        raise ResourceError("val300 must come from train and contain no leaderboard words")
    sel = set(get_split("select"))
    if sel & set(get_split("train")) or sel & benchmark_words() or sel & set(extra_answer_words(broad=True)):
        raise ResourceError("select must contain no training answers, extra answers or leaderboard words")
    textarena_accepted()
    extra_answer_words()
    extra_answer_words(broad=True)
    if set(get_split("train")) & (set(get_split("val")) | set(get_split("heldout"))):
        raise ResourceError("train overlaps val or heldout")
    if set(get_split("val")) & set(get_split("heldout")):
        raise ResourceError("val overlaps heldout")
    return man


@cache
def _split_cached(name: str) -> tuple[str, ...]:
    if name not in SPLIT_NAMES + BENCHMARK_NAMES + SELECTION_NAMES:
        raise KeyError(f"unknown split {name!r}; choose from {SPLIT_NAMES}")
    words = _read_words(f"splits/{name}.txt")
    want = manifest()["splits"][name]
    if len(words) != want["count"] or sha256_words(words) != want["sha256"]:
        raise ResourceError(f"split {name!r} does not match manifest.json; the benchmark files were changed")
    return tuple(words)


def get_split(name: str) -> list[str]:
    return list(_split_cached(name))


def benchmark_words() -> frozenset[str]:
    """Every word of every public leaderboard set: excluded from training answers in benchmark-clean runs."""
    return frozenset(w for name in BENCHMARK_NAMES for w in _split_cached(name))
