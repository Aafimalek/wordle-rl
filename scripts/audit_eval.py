"""Independent audit of saved Wordle evaluations.

Replays every game in a results.jsonl with its OWN feedback code (written separately from fiveletters.feedback,
using only the standard library) and checks that the recorded outcome is what the rules give:

* every guess the game accepted is a real 5-letter word from the official guess list;
* the recorded colours match the Wordle rule, including repeated letters;
* a game is won iff some accepted guess equals the answer, within 6 turns, and nothing is played after the win;
* under the retry rule (summary has "retries": K), a free turn is always a rejected non-word, at most K come in a
  row, and the 6-turn limit counts every other turn;
* under the leaderboard rule (summary has "open_vocab": true), an accepted guess outside the list must be five
  letters a-z and marked "nonword"; under every other rule no such guess is ever accepted;
* the win rate recomputed from scratch equals the one in summary.json;
* the answers are exactly the words of the split the summary names (no game dropped or added).

Usage:  python scripts/audit_eval.py <eval folder> [<eval folder> ...]
Exit code 1 if any check fails.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "src" / "fiveletters" / "resources"


def wordle_colours(guess: str, answer: str) -> tuple[int, ...]:
    """0 = grey, 1 = yellow, 2 = green. Greens first, then yellows from the letters still unused."""
    out = [0] * 5
    left = Counter(a for g, a in zip(guess, answer, strict=True) if g != a)
    for i, (g, a) in enumerate(zip(guess, answer, strict=True)):
        if g == a:
            out[i] = 2
    for i, g in enumerate(guess):
        if out[i] == 0 and left[g] > 0:
            out[i] = 1
            left[g] -= 1
    return tuple(out)


def _words(name: str) -> set[str]:
    return {w.strip().lower() for w in (RES / name).read_text().split() if len(w.strip()) == 5}


def audit(folder: Path, guess_list: set[str]) -> list[str]:
    problems = []
    summary = json.loads((folder / "summary.json").read_text())
    records = [json.loads(x) for x in (folder / "results.jsonl").read_text().splitlines() if x.strip()]
    wins = 0
    retries = int(summary.get("retries", 0))
    open_vocab = bool(summary.get("open_vocab", False))
    for r in records:
        ans, turns = r["answer"], r["turns"]
        won_at = None
        streak = used = 0
        for k, t in enumerate(turns):
            if won_at is not None:
                problems.append(f"{ans}: a turn was played after the win")
            if t.get("free"):
                streak += 1
                if t["valid"] or t["guess"] in guess_list or t.get("error") != "not_in_word_list":
                    problems.append(f"{ans}: a free turn that is not a rejected non-word ({t['guess']!r})")
                if streak > retries:
                    problems.append(f"{ans}: more than {retries} free rejections in a row")
            else:
                streak = 0
                used += 1
            if t["valid"]:
                g = t["guess"]
                five = len(g) == 5 and g.isascii() and g.isalpha() and g.islower()
                if g not in guess_list and not (open_vocab and five and t.get("nonword")):
                    problems.append(f"{ans}: accepted guess {g!r} is not in the official guess list")
                if t.get("nonword") and (g in guess_list or not open_vocab):
                    problems.append(f"{ans}: {g!r} is marked as a non-word but should not be")
                if tuple(t["feedback"]) != wordle_colours(g, ans):
                    problems.append(f"{ans}: wrong colours for {g!r}")
                if g == ans:
                    won_at = k
        if used > 6 or used != r.get("n_turns", used):
            problems.append(f"{ans}: {used} turns used (limit 6, recorded {r.get('n_turns')})")
        if (won_at is not None) != bool(r["won"]):
            problems.append(f"{ans}: recorded won={r['won']} but the replay says {won_at is not None}")
        wins += won_at is not None
    rate = wins / len(records) if records else 0.0
    if abs(rate - summary["win_rate"]) > 1e-4:
        problems.append(f"win rate {summary['win_rate']} in summary.json, {rate:.4f} by replay")
    split = summary.get("word_set")
    split_file = RES / "splits" / f"{split}.txt"
    if split_file.exists():
        expect = sorted(_words(f"splits/{split}.txt"))
        got = sorted(r["answer"] for r in records)
        if got != expect:
            problems.append(f"answers differ from split {split!r} ({len(got)} games vs {len(expect)} words)")
    print(f"{folder}: {len(records)} games, replayed win rate {rate:.4f} "
          f"({'OK' if not problems else f'{len(problems)} problem(s)'})")
    return problems


def main(argv: list[str]) -> int:
    guess_list = _words("answers.txt") | _words("extra_guesses.txt")
    bad = 0
    for f in argv:
        for p in audit(Path(f), guess_list):
            print("  -", p)
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
