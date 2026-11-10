"""Every recorded game of the final model (sft-gemma270m-v10-s1), alone and with word-level decoding, per set, for the
static demo. Test sets (wordlebench, jeffers, rare) were scored once; development sets (select, heldout) chose the model.
Each game is the exact record behind the paper's numbers."""
import json
import sys
from pathlib import Path

from fiveletters.prompts import parse_state

M = "hf-sft-gemma270m-v10-s1-best"
SETS = [("wordlebench", "final"), ("jeffers", "final"), ("rare", "final"), ("select", "v10"), ("heldout", "v10")]
MODES = [("alone", ""), ("word", "-constrained-wordle-word")]


def compact(r):
    turns = []
    for t in r["turns"]:
        fb = "".join("XYG"[f] for f in t["feedback"]) if t["valid"] else ""
        turns.append([(t.get("guess") or "")[:5], fb, parse_state(t.get("raw") or "") or ""])
    return {"w": int(r["won"]), "t": turns}


sets, summary = {}, {}
for split, kernel in SETS:
    sets[split] = {}
    for mode, suffix in MODES:
        f = Path("kaggle_runs") / kernel / "runs" / "eval" / split / (M + suffix) / "results.jsonl"
        recs = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
        won = [r for r in recs if r["won"]]
        summary[f"{split}:{mode}"] = {"games": len(recs), "won": len(won),
                                       "gw": round(sum(len(r["turns"]) for r in won) / max(len(won), 1), 2)}
        sets[split][mode] = {r["answer"]: compact(r) for r in sorted(recs, key=lambda r: r["answer"])}
out = {"model": "aafimalek2032/wordle-rl-gemma3-270m (sft-gemma270m-v10-s1)", "summary": summary, "sets": sets}
dest = Path(sys.argv[1]) / "games.json"
dest.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
print(dest.stat().st_size // 1024, "KB")
for k, v in summary.items():
    print(k, v, f"{100 * v['won'] / v['games']:.1f}%")
