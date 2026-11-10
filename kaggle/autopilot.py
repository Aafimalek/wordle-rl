"""Unattended overnight driver for the Kaggle runs. Deterministic: no AI, no open chat session.

Every 10 minutes it checks the tracked Kaggle jobs and, as each one ends:
  1. downloads its results (logs, eval summaries, metrics; never checkpoints);
  2. for the main session: if GRPO did not finish (paused at the time budget, or never started), pushes the
     same queue again (every command resumes from its Hub checkpoint); once GRPO has finished, launches
     the final scoring job (heldout, rare, TextArena) for the GRPO best model;
  3. rewrites kaggle_runs/MORNING_SUMMARY.md with every score found so far.

Run:  python kaggle/autopilot.py            (stop with Ctrl+C; safe to restart, state is on disk)
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = ROOT / "kaggle_runs"
STATE = RUNS / "autopilot_state_v4.json"
OWNER = "aafimalek"
H = "aafimalek2032/five-letters-runs"
V = "v4"  # round: job names fl-gemma-v4*, runs sft-gemma270m-v4 / grpo-gemma270m-v4
GRPO_RUN = f"grpo-gemma270m-{V}"
BEST = f"hub:{H}:{GRPO_RUN}/best"
SFT_BEST = f"hub:{H}:sft-gemma270m-{V}/best"  # SFT + constraint was the best heldout setting in round v3
GRPO_CMD = (f"grpo --config configs/grpo_gemma270m.toml --set run.name={GRPO_RUN} "
            f"--set model.init=hub:{H}:sft-gemma270m-{V}/best --set rollout.extra_answer_frac=0.5 "
            f"--set rollout.exclude_benchmark_answers=True --set checkpoint.hub_repo={H}")

PLANS = {
    # a GRPO resume needs only GRPO: SFT and its evals already ran and synced in the first session
    "grpo-resume": {"gpu0": [GRPO_CMD], "gpu1": []},
    # final scoring: unconstrained and dictionary-constrained, reported separately
    "final": {
        "gpu0": [f"eval --policy hf --model {BEST} --split heldout",
                 f"eval --policy hf --model {BEST} --split heldout --constrain wordle",
                 f"eval --policy hf --model {BEST} --split rare --constrain wordle",
                 f"eval --policy hf --model {BEST} --split full --constrain wordle",
                 f"eval --policy hf --model {BEST} --split wordlebench",
                 f"eval --policy hf --model {BEST} --split wordlebench --constrain wordle"],
        "gpu1": [f"eval --policy hf --model {BEST} --split rare",
                 f"public-eval --mode adapted --policy hf --model {BEST} --seeds 0:500",
                 f"public-eval --mode adapted --policy hf --model {BEST} --seeds 0:500 --constrain shared",
                 f"eval --policy hf --model {BEST} --split jeffers",
                 f"eval --policy hf --model {BEST} --split jeffers --constrain wordle"],
    },
}


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(RUNS / "autopilot.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def status(job: str) -> str:
    r = subprocess.run(["kaggle", "kernels", "status", f"{OWNER}/fl-{job}"], capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip().splitlines()
    return out[-1].split("status")[-1].strip(' "') if out else "UNKNOWN"


def job_py(*args) -> int:
    return subprocess.run([sys.executable, str(HERE / "job.py"), *args], cwd=ROOT).returncode


def push(name: str, plan_key: str) -> bool:
    plan = PLANS[plan_key]
    args = ["push", name, "--branch", "development", "--hours", "8.5", "--pip", "textarena", "--pip", "nltk"]
    for g in ("gpu0", "gpu1"):
        for c in plan[g]:
            args += [f"--{g}", c]
    ok = job_py(*args) == 0
    log(f"push fl-{name}: {'queued' if ok else 'refused (probably 2 sessions running); will retry in 10 min'}")
    return ok


def grpo_finished(job: str) -> bool:
    """True only if this session's GRPO command exited 0 (completed). Paused (75), failed or absent = False."""
    p = RUNS / job / "job_summary.json"
    if not p.exists():
        return False
    return any(r["cmd"].startswith("grpo") and r["rc"] == 0 for r in json.loads(p.read_text()).get("results", []))


def grpo_done_on_hub(job: str) -> bool:
    """GRPO's COMPLETED marker reaches the results dataset as soon as the command ends, even while the other
    GPU keeps the session alive. Checking it lets final scoring start hours earlier, in the free slot."""
    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download("aafimalek2032/five-letters-results", f"{job}/runs/{GRPO_RUN}/COMPLETED",
                        repo_type="dataset", force_download=True)
        return True
    except Exception:  # noqa: BLE001 - not there yet, or offline
        return False


def write_summary(state: dict) -> None:
    rows = []
    for p in sorted(RUNS.glob("*/runs/**/summary.json")):
        s = json.loads(p.read_text())
        rel = p.relative_to(RUNS).as_posix()
        if "all" in s:  # public environment
            for block in ("all", "clean"):
                b = s[block]
                if b.get("games"):
                    rows.append((rel, f"{s.get('public_env')} {s.get('mode')} [{block}]", b["games"], b["win_rate"],
                                 b.get("win_rate_ci95"), b["mean_guesses_fail7"], "-"))
        else:
            rows.append((rel, s.get("word_set"), s.get("games"), s.get("win_rate"), s.get("win_rate_ci95"),
                         s.get("mean_guesses_fail7"), s.get("invalid_turn_rate")))
    lines = [f"# Morning summary ({time.strftime('%Y-%m-%d %H:%M')})", "",
             "Statuses: " + ", ".join(f"fl-{j}: {status(j)}" for j in state["tracked"]), "",
             "| result | split / env | games | win rate | 95% CI | mean guesses (loss=7) | invalid turns |",
             "|---|---|---|---|---|---|---|"]
    lines += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} |" for r in rows]
    logf = RUNS / "autopilot.log"
    tail = logf.read_text(encoding="utf-8").splitlines()[-20:] if logf.exists() else []
    lines += ["", "## Autopilot log (last 20 lines)", "", "```", *tail, "```"]
    (RUNS / "MORNING_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(max_hours: float = 30.0) -> None:
    RUNS.mkdir(exist_ok=True)
    default = {"tracked": [f"gemma-{V}"], "grpo_jobs": [f"gemma-{V}"], "fetched": [], "pending": [],
               "resumes": 0, "final": None, "done": False}
    state = {**default, **(json.loads(STATE.read_text()) if STATE.exists() else {})}
    deadline = time.time() + max_hours * 3600
    log("autopilot started")
    while time.time() < deadline and not state["done"]:
        if not state.get("final_queued"):
            for job in state["grpo_jobs"]:
                if job not in state["fetched"] and grpo_done_on_hub(job):
                    log(f"GRPO completed in fl-{job} (marker on the Hub): final scoring now, in the free slot")
                    state["pending"].append(("final", f"gemma-{V}-final"))
                    state["final_queued"] = True
                    break
        for job in list(state["tracked"]):
            if job in state["fetched"]:
                continue
            st = status(job)
            if not any(k in st.upper() for k in ("COMPLETE", "ERROR", "CANCEL")):
                continue
            job_py("fetch", job)
            state["fetched"].append(job)
            log(f"fl-{job} ended ({st}); results fetched")
            if job in state["grpo_jobs"] and not state.get("final_queued"):
                if grpo_finished(job):
                    log("GRPO finished: final scoring next")
                    state["pending"].append(("final", f"gemma-{V}-final"))
                    state["final_queued"] = True
                elif state["resumes"] < 3:
                    state["resumes"] += 1
                    name = f"gemma-{V}-r{state['resumes']}"
                    log(f"GRPO not finished: resuming it as fl-{name}")
                    state["pending"].append(("grpo-resume", name))
                    state["grpo_jobs"].append(name)
                else:
                    log("GRPO still unfinished after 3 resumes; stopping relaunches for a human to look")
            if job == state["final"]:
                state["done"] = True
                log("final scoring fetched; autopilot done")
        still = []
        for plan_key, name in state["pending"]:
            if push(name, plan_key):
                state["tracked"].append(name)
                if plan_key == "final":
                    state["final"] = name
            else:
                still.append((plan_key, name))
        state["pending"] = still
        STATE.write_text(json.dumps(state, indent=2))
        write_summary(state)
        if not state["done"]:
            time.sleep(600)
    write_summary(state)


if __name__ == "__main__":
    main()
