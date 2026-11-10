"""Write one distillation round into kaggle/plans/distill.json, so every round uses the same, tested recipe.

    python kaggle/make_distill_plan.py --base sft-gemma270m-v6w-s1 --tag v6w-d1 --replay v6w --replay-frac 0.1

--base     the run whose best model plays the games (chosen on the select split)
--tag      names the data folder (data/traces/distill-<tag>) and the new run (sft-gemma270m-<tag>)
--replay   trace set mixed back in (its HF dataset is five-letters-traces-<replay>); the base model's own format
--replay-frac  share of the replay traces kept (deterministic subsample)
Launch: python kaggle/job.py plan kaggle/plans/distill.json distill-<tag>
"""

import argparse
import json
from pathlib import Path

H = "aafimalek2032/five-letters-runs"
W = " --constrain wordle --constrain-mode word"


def session(base: str, tag: str, replay: str, frac: float, hours: float = 6) -> dict:
    out = f"data/traces/distill-{tag}"
    b = f"hub:{H}:{base}/best"
    gen = lambda k: f"distill-data --model {b} --out {out} --openers model,slate --part {k} --parts 2"  # noqa: E731
    run = f"sft-gemma270m-{tag}"
    sft = (f"sft --config configs/sft_gemma270m.toml --set run.name={run} --set run.seed=1 --set model.name={b} "
           f"--set data.traces_dir={out} --set data.hub_dataset= "
           f"--set data.extra_traces_dir=data/traces/{replay} "
           f"--set data.extra_hub_dataset=aafimalek2032/five-letters-traces-{replay} "
           f"--set data.extra_sample_frac={frac} --set data.exclude_benchmark_answers=True "
           "--set train.epochs=1 --set train.lr=2e-05 --set train.warmup_steps=20 "
           "--set eval.split=select --set eval.games=534 --set eval.every_steps=100 --set eval.batch_size=100 "
           f"--set checkpoint.hub_repo={H}")
    m = f"hub:{H}:{run}/best"
    evals = [f"eval --policy hf --model {m} --split select", f"eval --policy hf --model {m} --split select{W}",
             f"eval --policy hf --model {m} --split heldout", f"eval --policy hf --model {m} --split heldout{W}"]
    gate = f"distill-{tag}-gen1"
    return {"name": f"distill-{tag}", "hours": hours, "pip": ["textarena", "nltk", "wordfreq"],
            "gpu0": [gen(0), f"gate {gate} --wait-minutes 240", sft, *evals],
            "gpu1": [gen(1), f"gate {gate} --mark"]}


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--base", required=True)
    a.add_argument("--tag", required=True)
    a.add_argument("--replay", required=True)
    a.add_argument("--replay-frac", type=float, default=0.1)
    a.add_argument("--file", default=str(Path(__file__).parent / "plans" / "distill.json"))
    args = a.parse_args()
    p = Path(args.file)
    plans = json.loads(p.read_text()) if p.exists() else {}
    plans[f"distill-{args.tag}"] = session(args.base, args.tag, args.replay, args.replay_frac)
    p.write_text(json.dumps(plans, indent=2) + "\n")
    print(f"wrote distill-{args.tag} into {p}")


if __name__ == "__main__":
    main()
