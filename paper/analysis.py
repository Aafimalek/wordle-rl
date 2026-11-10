"""Per-game analyses for the paper, computed from the raw game records (kaggle_runs/, not in git).

    python paper/analysis.py       # writes results/analysis/games.json and copies training curves to results/runs/

paper/build.py reads only committed files (results/), so the PDF can be rebuilt without the raw records.
"""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

from fiveletters.game import Turn, fits_clues

ROOT = Path(__file__).resolve().parents[1]
ANA = ROOT / "results" / "analysis"
M = "hf-sft-gemma270m-"
KERNEL = {"v7-s1": "v7", "v7s-s1": "v7", "v8-s1": "v8", "v8-s2": "v8", "v8-s3": "v9", "v9-s1": "v9",
          "v10-s1": "v10", "v10-s2": "v10"}
W = "-constrained-wordle-word"


def records(run: str, split: str = "select", suffix: str = "") -> list[dict] | None:
    f = ROOT / "kaggle_runs" / KERNEL[run] / "runs" / "eval" / split / f"{M}{run}-best{suffix}" / "results.jsonl"
    if not f.exists():
        return None
    return [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]


def loss_breakdown(recs: list[dict]) -> dict:
    lost = [r for r in recs if not r["won"]]
    nonword = fit = near = other_invalid = 0
    for r in recs:
        seen: list[Turn] = []
        for t in r["turns"]:
            g = t.get("guess") or ""
            if t["valid"]:
                seen.append(Turn("", g, True, tuple(t["feedback"])))
            elif len(g) == 5 and g.isalpha():
                nonword += 1
                fit += bool(fits_clues(g, seen)) if seen else 1
                near += sum(a != b for a, b in zip(g, r["answer"], strict=True)) == 1
            else:
                other_invalid += 1
    by_invalid = {}
    for k in (0, 1, 2, 3):
        games = [r for r in recs if (r["n_invalid"] == k if k < 3 else r["n_invalid"] >= 3)]
        by_invalid[str(k)] = {"games": len(games), "won": sum(r["won"] for r in games)}
    # Repeats: does the model write a rejected word again?
    first = repeats = next_valid = next_same = stuck = 0
    for r in recs:
        seen, ts = set(), r["turns"]
        for i, t in enumerate(ts):
            if t["valid"]:
                continue
            if t["guess"] in seen:
                repeats += 1
            else:
                first += 1
                if i + 1 < len(ts):
                    next_valid += ts[i + 1]["valid"]
                    next_same += (not ts[i + 1]["valid"]) and ts[i + 1]["guess"] == t["guess"]
            seen.add(t["guess"])
        bad = [t["guess"] for t in ts if not t["valid"]]
        stuck += (not r["won"]) and len(bad) >= 2 and len(set(bad)) == 1
    return {"games": len(recs), "lost": len(lost), "lost_with_invalid": sum(r["n_invalid"] > 0 for r in lost),
            "nonwords": nonword, "nonwords_fit_clues": fit, "nonwords_one_letter_off": near,
            "other_invalid_turns": other_invalid, "by_invalid_turns": by_invalid,
            "games_with_nonword": sum(r["n_invalid"] > 0 for r in recs), "first_time": first, "repeats": repeats,
            "next_valid": next_valid, "next_same": next_same, "lost_on_one_repeated": stuck}


def mcnemar(a: list[dict], b: list[dict]) -> dict:
    """Exact two-sided McNemar test on the same answers: does model b win games that model a loses?"""
    wa = {r["answer"]: r["won"] for r in a}
    wb = {r["answer"]: r["won"] for r in b}
    assert wa.keys() == wb.keys()
    gain = sum(wb[k] and not wa[k] for k in wa)
    loss = sum(wa[k] and not wb[k] for k in wa)
    n, m = gain + loss, min(gain, loss)
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(m + 1)) / 2 ** n) if n else 1.0
    return {"games": len(wa), "b_only": gain, "a_only": loss, "p": p}


def slim(r: dict) -> dict:
    return {"answer": r["answer"], "won": r["won"],
            "turns": [{"guess": t.get("guess") or "", "valid": t["valid"], "feedback": t.get("feedback")}
                      for t in r["turns"]]}


def examples(alone: list[dict], word: list[dict] | None) -> dict:
    """Deterministic picks: the first 4-turn clean win; the loss with the most clue-fitting non-words; and, with
    word-level decoding, the loss with the most guesses one letter from the answer. Ties go to the first answer."""
    out = {}
    wins = sorted((r for r in alone if r["won"] and r["n_turns"] == 4 and r["n_invalid"] == 0),
                  key=lambda r: r["answer"])
    if wins:
        out["win"] = slim(wins[0])

    def n_fit(r):
        seen, n = [], 0
        for t in r["turns"]:
            g = t.get("guess") or ""
            if t["valid"]:
                seen.append(Turn("", g, True, tuple(t["feedback"])))
            elif len(g) == 5 and g.isalpha() and (not seen or fits_clues(g, seen)):
                n += 1
        return n

    lost = sorted((r for r in alone if not r["won"]), key=lambda r: (-n_fit(r), r["answer"]))
    if lost:
        out["spelling_loss"] = slim(lost[0])
    if word:
        def n_near(r):
            return sum(sum(a != b for a, b in zip(t["guess"], r["answer"], strict=True)) == 1 for t in r["turns"])
        traps = sorted((r for r in word if not r["won"]), key=lambda r: (-n_near(r), r["answer"]))
        if traps:
            out["trap_loss"] = slim(traps[0])
    return out


def main():
    ANA.mkdir(parents=True, exist_ok=True)
    path = ANA / "games.json"
    out = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    out.setdefault("loss_breakdown", {})
    out.setdefault("paired", {})
    recs = {run: r for run in KERNEL if (r := records(run)) is not None}
    for run, r in recs.items():
        out["loss_breakdown"][run] = loss_breakdown(r)
    for a, b in (("v7-s1", "v8-s1"), ("v7-s1", "v8-s2"), ("v8-s1", "v8-s2"), ("v7-s1", "v8-s3"), ("v8-s2", "v9-s1"),
                 ("v8-s3", "v9-s1"), ("v8-s3", "v10-s1"), ("v8-s3", "v10-s2"), ("v10-s1", "v10-s2")):
        if a in recs and b in recs:
            out["paired"][f"{a}:{b}"] = mcnemar(recs[a], recs[b])
    if "v8-s1" in recs:
        out["examples"] = {"run": "v8-s1", **examples(recs["v8-s1"], records("v8-s1", suffix=W))}
    if "v10-s1" in recs and "examples" in out:
        # a recovery after a rejection: a won game whose first non-word is followed by a different, valid word
        def recovered(r):
            ts = r["turns"]
            i = next((k for k, t in enumerate(ts) if not t["valid"]), None)
            return r["won"] and i is not None and i + 1 < len(ts) and ts[i + 1]["valid"]
        rec = sorted((r for r in recs["v10-s1"] if recovered(r)), key=lambda r: (-r["n_turns"], r["answer"]))
        if rec:
            out["examples"]["recovered"] = slim(rec[0])
        # the same word as the "stuck" example, played by the final model (README picture: before and after)
        stuck = out["examples"].get("spelling_loss", {}).get("answer")
        same = [r for r in recs["v10-s1"] if r["answer"] == stuck]
        if same:
            out["examples"]["after_same_word"] = slim(same[0])
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")

    sessions = {}
    for f in sorted((ROOT / "kaggle_runs").glob("*/job_summary.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("minutes"):
            sessions[f.parent.name] = {"minutes": d["minutes"], "commit": (d.get("commit") or "")[:7]}
    if sessions:
        (ANA / "sessions.json").write_text(json.dumps(sessions, indent=1) + "\n", encoding="utf-8", newline="\n")

    copied = 0
    for run, kernel in KERNEL.items():
        src = ROOT / "kaggle_runs" / kernel / "runs" / f"sft-gemma270m-{run}" / "metrics.jsonl"
        if src.exists():
            dst = ROOT / "results" / "runs" / f"sft-gemma270m-{run}"
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst / "metrics.jsonl")
            copied += 1
    print(f"{len(recs)} runs analysed, {copied} training curves copied -> {path}")


if __name__ == "__main__":
    main()
