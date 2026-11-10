"""Evaluation harness: plays a policy on a frozen word set, resumably.

Free sessions die without warning, so the harness is built to be killed:

* every finished game is appended to ``results.jsonl`` and flushed to disk at once;
* on restart, finished words are skipped, and a half-written last line from a crash
  is detected and cut off;
* ``meta.json`` pins the protocol, word set checksum, rules and policy. Rerunning
  into the same folder with different settings is refused, so results never mix;
* a time budget or an exhausted API quota stops the run cleanly with
  ``EvalIncomplete``. Rerun the same command to continue.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .. import PROTOCOL_VERSION
from ..game import Game
from ..policies.base import Policy, QuotaExhausted
from ..prompts import parse_guess
from ..runtime import record_provenance
from ..words import load, sha256_words
from .metrics import summarize


class EvalIncomplete(RuntimeError):
    def __init__(self, done: int, total: int, reason: str):
        super().__init__(f"stopped after {done}/{total} games ({reason}); rerun the same command to resume")
        self.done, self.total, self.reason = done, total, reason


def play_games(policy: Policy, games: list[Game], valid_words=None) -> list[Game]:
    """Play games in lockstep: one batched policy call per turn across all unfinished games."""
    valid_words = valid_words or load().guess_set
    active = [g for g in games if not g.done]
    turn = 0
    while active:
        turn += 1
        t0 = time.monotonic()
        outs = policy.act(active)
        # heartbeat: a slow model and a hung one look the same without it
        print(f"[eval] turn {turn}: {len(active)} game(s) played this turn in {time.monotonic() - t0:.0f}s", flush=True)
        if len(outs) != len(active):
            raise RuntimeError(f"policy returned {len(outs)} outputs for {len(active)} games")
        for g, raw in zip(active, outs, strict=True):
            g.step(raw, parse_guess(raw), valid_words)
        active = [g for g in active if not g.done]
    return games


def _read_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    good: list[dict] = []
    keep_bytes = 0
    with open(path, "rb") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if not line.endswith(b"\n"):
                    raise ValueError("unterminated line")
            except ValueError:
                break  # truncated write from a crash: drop it and everything after
            good.append(rec)
            keep_bytes += len(line)
    if keep_bytes != path.stat().st_size:
        with open(path, "r+b") as f:
            f.truncate(keep_bytes)
    return good


def run_eval(policy: Policy, words: list[str], out_dir: str | Path, *, word_set: str, hard_mode: bool = False,
             batch_size: int = 64, time_budget_s: float | None = None, limit: int | None = None,
             retries: int = 0, open_vocab: bool = False) -> dict:
    if retries and open_vocab:
        raise ValueError("choose one rule: --retries or --open-vocab")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if limit:
        words = words[:limit]
    meta = {
        "protocol": PROTOCOL_VERSION,
        "word_set": word_set,
        "word_set_sha256": sha256_words(words),
        "n_words": len(words),
        "hard_mode": hard_mode,
        "max_turns": 6,
        "policy": policy.describe(),
    }
    if retries:  # the retry rule is a different protocol: label it, so it is never read as strict eval-v1
        meta["rule"] = f"retry-{retries}"
        meta["retries"] = retries
    if open_vocab:  # the leaderboards' own rule: no dictionary check
        meta["rule"] = "open-vocab"
        meta["open_vocab"] = True
    meta_path = out / "meta.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        if old != meta:
            diff = sorted(k for k in set(old) | set(meta) if old.get(k) != meta.get(k))
            raise RuntimeError(f"{out} holds results from different settings ({', '.join(diff)}). Use a new --out folder.")
    else:
        meta_path.write_text(json.dumps(meta, indent=2))
    # Kept apart from meta.json so a resume on another machine is allowed, but still recorded.
    record_provenance(out, job="eval")

    results_path = out / "results.jsonl"
    records = _read_results(results_path)
    done = {r["answer"] for r in records}
    todo = [w for w in words if w not in done]
    started = time.monotonic()
    if todo:
        print(f"[eval] {len(done)}/{len(words)} games already done, {len(todo)} to go -> {results_path}")
    try:
        with open(results_path, "a", encoding="utf-8", newline="\n") as f:
            for s in range(0, len(todo), batch_size):
                if time_budget_s and time.monotonic() - started > time_budget_s:
                    raise EvalIncomplete(len(records), len(words), "time budget reached")
                games = [Game(w, hard_mode=hard_mode, retries=retries, open_vocab=open_vocab) for w in todo[s : s + batch_size]]
                try:
                    play_games(policy, games)
                except QuotaExhausted as e:
                    raise EvalIncomplete(len(records), len(words), str(e)) from e
                for g in games:
                    rec = g.record()
                    f.write(json.dumps(rec) + "\n")
                    records.append(rec)
                f.flush()
                os.fsync(f.fileno())
                print(f"[eval] {len(records)}/{len(words)} games, win rate so far {sum(r['won'] for r in records) / len(records):.3f}")
    except EvalIncomplete:
        (out / "summary.partial.json").write_text(json.dumps({**meta, **summarize(records)}, indent=2))
        raise
    finally:
        policy.close()

    summary = {**meta, **summarize(records)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
