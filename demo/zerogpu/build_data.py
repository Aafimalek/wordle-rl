"""Bundle the recorded final-test games (WordleBench, Jeffers; alone and word-level) for the demo Space."""
import json
import sys
from pathlib import Path

from fiveletters.prompts import parse_state

src = Path("kaggle_runs/final/runs/eval")
out = {}
for split in ("wordlebench", "jeffers"):
    for suffix, mode in (("", "alone"), ("-constrained-wordle-word", "word")):
        f = src / split / f"hf-sft-gemma270m-v10-s1-best{suffix}" / "results.jsonl"
        games = []
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            games.append({"answer": r["answer"], "won": r["won"],
                          "turns": [{"guess": t.get("guess") or "", "valid": t["valid"], "feedback": t.get("feedback"),
                                     "state": parse_state(t.get("raw") or "") or ""} for t in r["turns"]]})
        out[f"{split}:{mode}"] = sorted(games, key=lambda g: g["answer"])
        print(split, mode, len(games), sum(g["won"] for g in games))
dest = Path(sys.argv[1])
dest.mkdir(parents=True, exist_ok=True)
(dest / "final_test_games.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
print((dest / "final_test_games.json").stat().st_size, "bytes")
