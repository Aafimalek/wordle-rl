"""Round v10: recovery from the model's OWN rejected words (one DAgger-style round).

Diagnosis (PAPER_NOTES, 2026-11-10): when v8 writes a non-word and the game rejects it, the next turn is the same
non-word again in 88 of 89 cases (select, alone), and 83 of its 95 lost games end on one repeated non-word. The drill
did show rejected words, but they were random near-misses, never the word the model itself believes in.

Here the model plays alone, under the strict rule, on training-side hidden words. Every first rejection of a non-word
that still leaves a turn becomes training boards: the game exactly as the model played it, up to and including its
rejected word, with a real word that fits the clues as the target.

Fairness, as for the drill: hidden words are never dev or test answers (`recovery_hidden_words`). Targets are common
public words that fit the clues (dev and test answers included, the disclosed choice of v7 and v8), or the hidden
word itself.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from .drill import DRILL_ZIPF_MIN, _flush, dev_test_words, write_manifest
from .eval.harness import play_games
from .game import Game, fits_clues
from .prompts import DEFAULT_FORMAT, PromptFormat, render_completion, render_prompt
from .words import extra_answer_words, get_split, load


def recovery_hidden_words(limit: int | None = None) -> list[str]:
    """Training-side words the model has not memorised as answers: every valid guess that is not a dev or test answer
    and not a train or extra answer, most common first (public `wordfreq` frequencies), ties by spelling."""
    from .solvers import word_zipf

    skip = dev_test_words() | set(get_split("train")) | set(extra_answer_words())
    words = sorted(w for w in load().guess_set if w not in skip)
    zipf = word_zipf(words)
    order = sorted(range(len(words)), key=lambda i: (-float(zipf[i]), words[i]))
    out = [words[i] for i in order]
    return out[:limit] if limit else out


def common_targets() -> list[str]:
    """The drill's target list: every common public word."""
    from .solvers import word_zipf

    words = sorted(load().guess_set)
    zipf = word_zipf(words)
    return [w for w, z in zip(words, zipf, strict=True) if z >= DRILL_ZIPF_MIN]


def _is_nonword(turn, guess_set) -> bool:
    g = turn.guess or ""
    return (not turn.valid) and len(g) == 5 and g.isascii() and g.isalpha() and g.islower() and g not in guess_set


def recovery_samples(game: Game, game_id: int, targets: list[str], fmt: PromptFormat, rng: random.Random,
                     per_event: int = 4, repeat_prob: float = 0.3) -> tuple[list[dict], int]:
    """Samples for one played game, and the number of rejection events used.

    An event is the first time a given non-word is rejected, if a turn is left after it. The board is the game up to
    and including that turn. With probability `repeat_prob` (and a spare turn) the same word is rejected twice, as
    happens when the model repeats itself. Targets: up to `per_event` different words that fit every clue so far and
    were not played, drawn from `targets` plus the hidden word.
    """
    guess_set = load().guess_set
    if game.answer in dev_test_words():
        raise ValueError(f"{game.answer} is a dev or test answer; recovery games use training-side words only")
    rows, events, seen_bad = [], 0, set()
    for i, t in enumerate(game.turns):
        if not _is_nonword(t, guess_set) or t.guess in seen_bad:
            continue
        seen_bad.add(t.guess)
        board = Game(game.answer, hard_mode=game.hard_mode)
        for u in game.turns[:i + 1]:
            board.step(u.raw, u.guess, guess_set, by_model=u.by_model)
        if board.done:
            continue
        if board.turns_left >= 2 and rng.random() < repeat_prob:
            board.step(t.raw, t.guess, guess_set, by_model=t.by_model)
        played = {u.guess for u in board.turns}
        valid = board.valid_turns
        fit = [w for w in targets if w not in played and fits_clues(w, valid)]
        if game.answer not in fit:
            fit.append(game.answer)
        events += 1
        prompt = render_prompt(board, fmt)
        for w in rng.sample(fit, min(per_event, len(fit))):
            rows.append({"prompt": prompt, "completion": render_completion(board, w, fmt), "answer": game.answer,
                         "game": game_id, "turn": len(board.turns) + 1, "game_won": True, "source": "recovery",
                         "rejected": t.guess})
    return rows, events


def generate(policy, out_dir: str | Path, *, limit: int | None = 4000, part: int = 0, parts: int = 1,
             per_event: int = 4, repeat_prob: float = 0.3, seed: int = 1, shard_size: int = 5000,
             chunk: int = 500, fmt: PromptFormat | None = None) -> dict:
    """Play hidden words i with i % parts == part, alone and strict, and write the recovery samples as a trace
    folder (`shard-recovery-*.jsonl`, `games.jsonl`, `trace_config.json`, `manifest.json`)."""
    fmt = fmt or getattr(policy, "fmt", None) or DEFAULT_FORMAT
    out = Path(out_dir)
    if any(out.glob("shard-*.jsonl")):
        raise FileExistsError(f"{out} already holds shards; build into an empty folder")
    out.mkdir(parents=True, exist_ok=True)
    words = [w for i, w in enumerate(recovery_hidden_words(limit)) if i % parts == part]
    targets = common_targets()
    guess_set = load().guess_set
    rng = random.Random(int.from_bytes(hashlib.sha256(f"recovery|{seed}|{part}|{parts}".encode()).digest()[:8], "big"))
    rows, shard, written, events, won, with_nonword, repeats, first = [], 0, 0, 0, 0, 0, 0, 0
    with open(out / "games.jsonl", "w", encoding="utf-8") as log:
        for start in range(0, len(words), chunk):
            games = [Game(w) for w in words[start:start + chunk]]
            play_games(policy, games, guess_set)
            for k, g in enumerate(games):
                log.write(json.dumps(g.record()) + "\n")
                won += g.won
                bad = [t.guess for t in g.turns if _is_nonword(t, guess_set)]
                with_nonword += bool(bad)
                first += len(set(bad))
                repeats += len(bad) - len(set(bad))
                new, n = recovery_samples(g, start + k, targets, fmt, rng, per_event, repeat_prob)
                events += n
                rows += new
            while len(rows) >= shard_size:
                written += _flush(out, f"shard-recovery-{shard:05d}.jsonl", rows[:shard_size])
                rows, shard = rows[shard_size:], shard + 1
            print(f"[recovery] {min(start + chunk, len(words))}/{len(words)} games, {events} events", flush=True)
    if rows:
        written += _flush(out, f"shard-recovery-{shard:05d}.jsonl", rows)
    if not written:
        raise RuntimeError("the model produced no rejected word with a turn left; nothing to train on")
    stats = {"games": len(words), "won": won, "games_with_nonword": with_nonword, "nonwords_first_time": first,
             "nonwords_repeated": repeats, "events": events, "samples": written}
    (out / "trace_config.json").write_text(json.dumps(
        {"source": "v10-recovery", **fmt.to_dict(), "policy": policy.describe(), "limit": limit, "part": part,
         "parts": parts, "per_event": per_event, "repeat_prob": repeat_prob, "seed": seed,
         "zipf_min": DRILL_ZIPF_MIN, **stats}, indent=2))
    write_manifest(out)
    print(f"[recovery] {stats}", flush=True)
    return stats
