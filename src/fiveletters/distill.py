"""Step 2.1 of the roadmap: distil word-level constrained decoding into the model's weights (expert iteration).

The current model plays games WITH word-level decoding (it picks the real word it rates highest; the code never looks
at the clues). Every model turn of every WON game becomes an SFT sample: the prompt is the board before the turn and the
completion is the model's own reply (its <state> line plus the real word it chose). Fine-tuning on these teaches the
model to type that real word itself, so it needs the decoding helper less.

Only training answers are played: the train split and the extra (non-test) answers, minus every leaderboard word. The
selection split (`select`), heldout, val, WordleBench and Jeffers are never played here. Output has the trace format
(`shard-*.jsonl` + `manifest.json`), so `fiveletters sft` trains on it unchanged. Shards are written atomically and
skipped when present, so a stopped run resumes; `part`/`parts` split the work across GPUs.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .eval.harness import play_games
from .game import Game
from .prompts import DEFAULT_FORMAT, PromptFormat, render_completion, render_prompt
from .words import benchmark_words, extra_answer_words, get_split, load


def distill_answers() -> list[str]:
    """Training answers only: train split + extra answers, minus leaderboard words. Never dev or test words."""
    bench = benchmark_words()
    never = set(get_split("select")) | set(get_split("heldout")) | set(get_split("val")) | bench
    words = sorted((set(get_split("train")) | set(extra_answer_words())) - never)
    return words


def samples_from_game(game: Game, game_id: int, fmt: PromptFormat = DEFAULT_FORMAT) -> list[dict]:
    """SFT samples for the model's own turns of a won game (forced opener turns are context, not targets)."""
    if not game.won:
        return []
    guess_set = load().guess_set
    replay = Game(game.answer, hard_mode=game.hard_mode)
    out = []
    for k, t in enumerate(game.turns, 1):
        if t.by_model and t.valid:
            out.append({"prompt": render_prompt(replay, fmt), "completion": t.raw, "answer": game.answer,
                        "game": game_id, "turn": k, "game_won": True, "source": "distill"})
        replay.step(t.raw, t.guess, guess_set, by_model=t.by_model)
    return out


def generate(policy, out_dir: str | Path, *, openers: tuple[str | None, ...] = (None, "slate"), shard_size: int = 200,
             part: int = 0, parts: int = 1, limit: int | None = None, time_budget_s: float | None = None,
             fmt: PromptFormat | None = None) -> dict:
    """Play the distillation games for shards i with i % parts == part. Returns a progress summary."""
    fmt = fmt or getattr(policy, "fmt", None) or DEFAULT_FORMAT  # the policy's own prompt format
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    words = distill_answers()[:limit] if limit else distill_answers()
    shards = [words[i:i + shard_size] for i in range(0, len(words), shard_size)]
    guess_set = load().guess_set
    t0 = time.monotonic()
    done = skipped = 0
    for i, chunk in enumerate(shards):
        if i % parts != part:
            continue
        path = out / f"shard-{i:05d}.jsonl"
        if path.exists():
            skipped += 1
            continue
        if time_budget_s is not None and time.monotonic() - t0 > time_budget_s:
            print(f"[distill] time budget reached; {done} shard(s) written, rerun to continue", flush=True)
            return {"complete": False, "paused": True, "written": done, "skipped": skipped}
        games, ids = [], []
        for w in chunk:
            for opener in openers:
                g = Game(w)
                if opener:
                    g.step(render_completion(g, opener, fmt), opener, guess_set, by_model=False)
                games.append(g)
                ids.append(i * shard_size * len(openers) + len(ids))
        play_games(policy, games, guess_set)
        rows = [s for g, gid in zip(games, ids, strict=True) for s in samples_from_game(g, gid, fmt)]
        won = sum(g.won for g in games)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        os.replace(tmp, path)
        done += 1
        print(f"[distill] {path.name}: {won}/{len(games)} games won -> {len(rows)} samples", flush=True)
    present = sorted(out.glob("shard-*.jsonl"))
    complete = len(present) == len(shards)
    if complete:
        total = sum(1 for p in present for _ in open(p, encoding="utf-8"))
        manifest = {"shards": len(shards), "samples": total,
                    "config": {"source": "distill", "openers": [o or "model" for o in openers],
                               "policy": policy.describe(), "answers": len(words), "shard_size": shard_size}}
        # same metadata file as `fiveletters traces`, so `fiveletters sft` reads the prompt format from it
        (out / "trace_config.json").write_text(json.dumps(
            {"source": "distill", "spaced": fmt.spaced, "state_line": fmt.state_line, "guess_word": fmt.guess_word,
             **manifest["config"]}, indent=2))
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"[distill] complete: {len(shards)} shards, {total} samples", flush=True)
    return {"complete": complete, "paused": False, "written": done, "skipped": skipped}
