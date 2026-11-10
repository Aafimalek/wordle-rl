"""Round v6 data helpers: a spelling drill, and conversion of spaced traces to the whole-word guess format.

Spelling drill. The model alone loses mostly by inventing near-miss words (ABONE for ATONE): it knows the pattern but
cannot produce a real word that fits. Each drill sample is a board after 1-3 ordinary opening guesses, and the target
is a real dictionary word that fits every clue, picked by general English frequency (the most common one half the
time, otherwise a random common one). Rules that keep it fair, stated in the paper:

* the boards are built around hidden words drawn only from training-side words: common dictionary words that are
  NOT in val, heldout, select, rare, WordleBench or Jeffers. No drill board is ever built around a dev or test word;
* the targets may be any real word from the public 12,972-word guess list, test answers included, because knowing
  English words is general knowledge (any speaker or frontier model has it). Frequencies come from the public
  `wordfreq` package, never from Wordle answer statistics.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from pathlib import Path

import numpy as np

from .game import Game
from .prompts import PromptFormat, render_completion, render_prompt
from .words import benchmark_words, get_split, guess_vocab, load

DRILL_ZIPF_MIN = 2.0  # "common": at least this frequency on the Zipf scale (saute 2.5, crame 1.3, axone 0.0)


def drill_hidden_pool(zipf: np.ndarray, words: list[str]) -> list[str]:
    never = (set(get_split("val")) | set(get_split("heldout")) | set(get_split("select")) | set(get_split("rare"))
             | benchmark_words())
    return [w for w, z in zip(words, zipf, strict=True) if z >= DRILL_ZIPF_MIN and w not in never]


def generate_drill(out_dir: str | Path, n_samples: int, fmt: PromptFormat, seed: int = 1, shard_size: int = 5000,
                   prefix: str = "shard-drill") -> dict:
    from .solvers import CommonProbeSolver

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    solver = CommonProbeSolver(pool="all", vocab="wordle", common=True, probe=False)  # candidates over 12,972 words
    words = solver.guesses
    zipf = solver.zipf
    hidden_pool = drill_hidden_pool(zipf, words)
    openers = [w for w, z in zip(words, zipf, strict=True) if z >= 3.5 and w in set(guess_vocab("shared"))]
    guess_set = load().guess_set
    rng = random.Random(int.from_bytes(hashlib.sha256(f"drill|{seed}".encode()).digest()[:8], "big"))
    rows, written = [], 0
    while written + len(rows) < n_samples:
        hidden = rng.choice(hidden_pool)
        g = Game(hidden)
        for _ in range(rng.choice((1, 2, 3))):
            w = rng.choice(openers)
            g.step(render_completion(g, w, fmt), w, guess_set, by_model=False)
            if g.done:
                break
        if g.done:
            continue
        cands = solver.candidates(g.turns)
        common = [int(c) for c in cands if zipf[c] >= DRILL_ZIPF_MIN and words[c] not in {t.guess for t in g.turns}]
        if not common:
            continue
        if rng.random() < 0.5:
            target = words[max(common, key=lambda c: zipf[c])]
        else:
            target = words[rng.choice(common)]
        rows.append({"prompt": render_prompt(g, fmt), "completion": render_completion(g, target, fmt),
                     "answer": hidden, "game": -1, "turn": len(g.turns) + 1, "game_won": True, "source": "drill"})
        if len(rows) == shard_size:
            written += _flush(out, f"{prefix}-{written // shard_size:05d}.jsonl", rows)
            rows = []
    if rows:
        written += _flush(out, f"{prefix}-{written // shard_size:05d}.jsonl", rows)
    return {"samples": written, "hidden_pool": len(hidden_pool), "zipf_min": DRILL_ZIPF_MIN, "seed": seed}


def _flush(out: Path, name: str, rows: list[dict]) -> int:
    tmp = out / (name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, out / name)
    return len(rows)


_SPACED_GUESS = re.compile(r"<guess>([A-Z](?: [A-Z]){4})</guess>")


def to_guess_word(text: str) -> str:
    """'<guess>C R A T E</guess>' -> '<guess>crate</guess>' (every occurrence: targets and the instruction example).
    The board history ('Turn 1: C R A T E -> ...') keeps its spaced letters."""
    return _SPACED_GUESS.sub(lambda m: "<guess>" + m.group(1).replace(" ", "").lower() + "</guess>", text)


def convert_traces(src: str | Path, dst: str | Path) -> dict:
    """Copy a spaced trace folder to the whole-word guess format. Same games, same order, same counts."""
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((src / "trace_config.json").read_text())
    if not cfg.get("spaced", False) or cfg.get("guess_word", False):
        raise ValueError(f"{src} is not a spaced-letter trace folder")
    total = 0
    for p in sorted(src.glob("shard-*.jsonl")):
        tmp = dst / (p.name + ".tmp")
        with open(p, encoding="utf-8") as f, open(tmp, "w", encoding="utf-8", newline="\n") as g:
            for line in f:
                s = json.loads(line)
                s["prompt"], s["completion"] = to_guess_word(s["prompt"]), to_guess_word(s["completion"])
                g.write(json.dumps(s) + "\n")
                total += 1
        os.replace(tmp, dst / p.name)
    (dst / "trace_config.json").write_text(json.dumps({**cfg, "guess_word": True, "converted_from": str(src)},
                                                      indent=2))
    write_manifest(dst)
    return {"samples": total}


def write_manifest(folder: str | Path) -> dict:
    folder = Path(folder)
    shards = sorted(folder.glob("shard-*.jsonl"))
    total = sum(1 for p in shards for _ in open(p, encoding="utf-8"))
    cfg = json.loads((folder / "trace_config.json").read_text())
    m = {"shards": len(shards), "samples": total, "config": cfg}
    (folder / "manifest.json").write_text(json.dumps(m, indent=2))
    return m


# ---------------------------------------------------------------------------------------------------------------
# Round v7: a BALANCED drill. Diagnosis (PAPER_NOTES, 2026-11-10): the model alone wins 99% of the games in which it
# never types a non-word, types almost only words it typed in training, and the v6 drill gave a median of 5 targets
# per word. Here every common word is the target the same number of times.
# ---------------------------------------------------------------------------------------------------------------

BALANCED_DEPTHS = ((1, 0.30), (2, 0.35), (3, 0.25), (4, 0.10))  # guesses on the board before the target (v7)
# v8: the model's invalid turns are at turns 4-6 (v7 on select: 28 / 52 / 60%), so most boards are deep and some have
# five guesses (target at turn 6, never drilled in v7).
DEEP_DEPTHS = ((1, 0.10), (2, 0.20), (3, 0.30), (4, 0.25), (5, 0.15))


def dev_test_words() -> set[str]:
    """Every word that is an answer in a dev or test set. No drill board is ever built around one of these."""
    return (set(get_split("val")) | set(get_split("heldout")) | set(get_split("select")) | set(get_split("rare"))
            | benchmark_words())


def _near_miss(word: str, rng: random.Random, guess_set) -> str:
    """A non-word one letter away from `word` (the model's typical mistake: ABONE for ATONE)."""
    while True:
        i = rng.randrange(5)
        junk = word[:i] + rng.choice("abcdefghijklmnopqrstuvwxyz") + word[i + 1:]
        if junk not in guess_set:
            return junk


def generate_balanced_drill(out_dir: str | Path, per_word: int, fmt: PromptFormat, seed: int = 1,
                            shard_size: int = 5000, prefix: str = "shard-drill", strict: bool = False,
                            invalid_prob: float = 0.15, limit_words: int | None = None,
                            depths: tuple = BALANCED_DEPTHS, max_invalid: int = 1) -> dict:
    """`per_word` samples for every common word (Zipf >= DRILL_ZIPF_MIN): a board the word fits -> that word.

    A board is an opener (CRATE 60% of the time) followed by 0-3 common words that each fit the clues so far, as a
    consistent player would guess. It is kept only if at least one training-side word (`drill_hidden_pool`) also fits
    it; that word is the hidden word the board is built around, so no board is built around a dev or test word.
    Targets may be any common public word, dev and test answers included (disclosed in the paper). With
    `strict=True` no dev or test answer is ever a target: that variant measures how far the drill carries over to
    words it never showed. Some boards contain rejected non-words, to teach recovery: one with probability
    `invalid_prob`; with `max_invalid` > 1, each further one again with probability `invalid_prob`, as long as the
    board still leaves a turn for the target (6 turns in all).
    """
    from .solvers import CommonProbeSolver

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    solver = CommonProbeSolver(pool="all", vocab="wordle", common=True, probe=False)
    words, zipf, M = solver.guesses, solver.zipf, solver.M
    n = len(words)
    never = dev_test_words()
    common_mask = zipf >= DRILL_ZIPF_MIN
    hidden_mask = np.array([bool(common_mask[i]) and words[i] not in never for i in range(n)])
    targets = [int(i) for i in np.flatnonzero(common_mask) if not (strict and words[i] in never)]
    if limit_words:
        targets = targets[:limit_words]
    shared = set(guess_vocab("shared"))
    openers = [int(i) for i in np.flatnonzero(zipf >= 3.5) if words[i] in shared]
    crate = solver.gidx["crate"]
    guess_set = load().guess_set
    rng = random.Random(int.from_bytes(hashlib.sha256(f"balanced-drill|{seed}|{strict}".encode()).digest()[:8], "big"))
    if max(d for d, _ in depths) > 5 or max_invalid < 1:
        raise ValueError("a board holds at most 5 guesses, and max_invalid must be at least 1")
    depths, weights = zip(*depths, strict=True)

    def board_for(w: int):
        for _ in range(30):
            fit, hid, gs = np.ones(n, dtype=bool), hidden_mask, []
            for d in range(rng.choices(depths, weights)[0]):
                if d == 0:
                    g = crate if rng.random() < 0.6 else rng.choice(openers)
                else:
                    pool = np.flatnonzero(fit & common_mask)
                    pool = pool[pool != w]
                    if len(pool) == 0:
                        break
                    g = int(pool[rng.randrange(len(pool))])
                if g == w:
                    break
                same = M[g] == M[g, w]
                new_hid = hid & same
                if not new_hid.any():  # no training-side word shares this board: stop one guess earlier
                    break
                fit, hid = fit & same, new_hid
                gs.append(g)
            if gs:
                left = np.flatnonzero(hid)
                return gs, int(left[rng.randrange(len(left))])
        return None

    rows, written, skipped, depth_hist, shard = [], 0, 0, {}, 0
    for _ in range(per_word):
        order = targets[:]
        rng.shuffle(order)
        for w in order:
            b = board_for(w)
            if b is None:
                skipped += 1
                continue
            gs, h = b
            g = Game(words[h])
            if max_invalid == 1:  # v7: the same random draws as before, so v7 data is reproduced exactly
                bad = [rng.randrange(1, len(gs) + 1)] if rng.random() < invalid_prob and len(gs) < 5 else []
            else:
                bad = []
                while len(bad) < min(max_invalid, 5 - len(gs)) and rng.random() < invalid_prob:
                    bad.append(rng.randrange(1, len(gs) + 1))
            for k, gi in enumerate(gs):
                g.step(render_completion(g, words[gi], fmt), words[gi], guess_set, by_model=False)
                for _ in range(bad.count(k + 1)):
                    junk = _near_miss(words[w] if rng.random() < 0.5 else words[rng.choice(targets)], rng, guess_set)
                    g.step(render_completion(g, junk, fmt), junk, guess_set, by_model=False)
            if g.done:  # cannot happen: the hidden word is never one of the guesses
                raise RuntimeError("drill board ended the game")
            depth_hist[len(gs)] = depth_hist.get(len(gs), 0) + 1
            rows.append({"prompt": render_prompt(g, fmt), "completion": render_completion(g, words[w], fmt),
                         "answer": words[h], "game": -1, "turn": len(g.turns) + 1, "game_won": True,
                         "source": "drill"})
            if len(rows) == shard_size:
                written += _flush(out, f"{prefix}-{shard:05d}.jsonl", rows)
                rows, shard = [], shard + 1
    if rows:
        written += _flush(out, f"{prefix}-{shard:05d}.jsonl", rows)
    return {"samples": written, "targets": len(targets), "per_word": per_word, "skipped": skipped, "strict": strict,
            "hidden_pool": int(hidden_mask.sum()), "zipf_min": DRILL_ZIPF_MIN, "invalid_prob": invalid_prob,
            "seed": seed, "depths": {str(k): v for k, v in sorted(depth_hist.items())},
            "depth_weights": {str(d): w for d, w in zip(depths, weights, strict=True)}, "max_invalid": max_invalid}


def copy_replay(src: str | Path, dst: str | Path, frac: float, prefix: str, shard_size: int = 5000) -> dict:
    """Mix a deterministic share of an existing trace set (teacher games) into `dst`, so the drill does not wash out
    the game strategy. Games whose answer is a leaderboard word are dropped here as well as at training time."""
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((src / "trace_config.json").read_text())
    if not cfg.get("spaced", False) or cfg.get("guess_word", False) or not cfg.get("state_line", False):
        raise ValueError(f"{src} is not in the spaced-letter format with a state line")
    bench = benchmark_words()
    rows, written, shard = [], 0, 0
    for p in sorted(src.glob("shard-*.jsonl")):
        with open(p, encoding="utf-8") as f:
            for line in f:
                s = json.loads(line)
                key = f"{s['answer']}|{s.get('game')}|{s.get('turn')}|{s['completion']}".encode()
                if s["answer"] in bench or int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64 >= frac:
                    continue
                rows.append(s)
                if len(rows) == shard_size:
                    written += _flush(dst, f"{prefix}-{shard:05d}.jsonl", rows)
                    rows, shard = [], shard + 1
    if rows:
        written += _flush(dst, f"{prefix}-{shard:05d}.jsonl", rows)
    return {"samples": written, "from": src.name, "frac": frac, "teacher": cfg.get("teacher")}


def build_v7(out_dir: str | Path, per_word: int, replay: list[tuple[str, float]], seed: int = 1,
             strict: bool = False, limit_words: int | None = None, deep: bool = False, invalid_prob: float = 0.15,
             max_invalid: int = 1) -> dict:
    """One training folder: the balanced drill plus replayed teacher games. `replay` = [(trace folder, share), ...]."""
    out = Path(out_dir)
    if any(out.glob("shard-*.jsonl")):
        raise FileExistsError(f"{out} already holds shards; build into an empty folder")
    fmt = PromptFormat(spaced=True, state_line=True, guess_word=False)
    drill = generate_balanced_drill(out, per_word, fmt, seed=seed, strict=strict, limit_words=limit_words,
                                    depths=DEEP_DEPTHS if deep else BALANCED_DEPTHS, invalid_prob=invalid_prob,
                                    max_invalid=max_invalid)
    reps = [copy_replay(src, out, frac, f"shard-replay{k}") for k, (src, frac) in enumerate(replay)]
    (out / "trace_config.json").write_text(json.dumps(
        {"source": "v8" if deep else "v7", **fmt.to_dict(), "drill": drill, "replay": reps}, indent=2))
    m = write_manifest(out)
    return {"samples": m["samples"], "drill": drill, "replay": reps}
