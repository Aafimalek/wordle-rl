"""Run GPU jobs on Kaggle from your laptop (free T4 x2), without opening the website.

One-time setup (tokens are read from files, never typed into commands):
    ~/.kaggle/access_token              Kaggle API token
    ~/.five-letters-secrets/hf_token    Hugging Face token (write, plus gated-repo read for Gemma)
    ~/.five-letters-secrets/github_token  GitHub fine-grained token, read-only Contents on this repo
    python kaggle/job.py secrets        uploads both tokens to a PRIVATE Kaggle dataset
    python kaggle/job.py cache          uploads the solver caches (saves minutes per session)

Jobs (each GPU runs its commands in order, the two GPUs in parallel):
    python kaggle/job.py push sft1 --gpu0 "sft --config ..." --gpu1 "eval ..." --gpu1 "eval ..."
    python kaggle/job.py status sft1
    python kaggle/job.py fetch sft1       logs, eval results and metrics (not checkpoints)

To resume a paused job, push the same commands again: every command resumes by itself.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
def _detect_owner() -> str:
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    try:
        out = subprocess.run(["kaggle", "config", "view"], capture_output=True, text=True, timeout=60).stdout
        for line in out.splitlines():
            if line.strip().startswith("- username:"):
                return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "aafimalek"


OWNER = _detect_owner()
REPO = "https://github.com/Aafimalek/rl-environments-startup-guide.git"
SUBDIR = "projects/five-letters"
SECRETS_DIR = Path.home() / ".five-letters-secrets"
SECRETS_DATASET = f"{OWNER}/five-letters-secrets"
CACHE_DATASET = f"{OWNER}/five-letters-cache"
FETCH_PATTERN = r"(logs/|env/|job_summary\.json|runs/eval/|runs/public/|metrics\.jsonl|config\.json|provenance\.jsonl|COMPLETED)"


def kaggle(*args, check=True) -> subprocess.CompletedProcess:
    r = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args[:2])} failed:\n{r.stdout}\n{r.stderr}")
    return r


def slug(name: str) -> str:
    return f"fl-{name}".lower().replace("_", "-")


def _upload_dataset(dataset_id: str, files: list[Path], title: str, message: str) -> None:
    with tempfile.TemporaryDirectory() as d:
        for f in files:
            shutil.copy(f, d)
        meta = {"title": title, "id": dataset_id, "licenses": [{"name": "other"}]}
        Path(d, "dataset-metadata.json").write_text(json.dumps(meta))
        exists = kaggle("datasets", "status", dataset_id, check=False).returncode == 0
        if exists:
            kaggle("datasets", "version", "-p", d, "-m", message, "-q")
        else:
            kaggle("datasets", "create", "-p", d, "-q")  # private unless --public is passed
    print(f"uploaded {len(files)} file(s) to private dataset {dataset_id}")


def cmd_secrets(_a) -> None:
    files = [SECRETS_DIR / "hf_token", SECRETS_DIR / "github_token"]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"missing: {missing}")
    _upload_dataset(SECRETS_DATASET, files, "five-letters-secrets", "update tokens")


def cmd_cache(_a) -> None:
    cache = Path(os.environ.get("FL_CACHE_DIR") or Path.home() / ".cache" / "fiveletters")
    files = sorted(cache.glob("patterns-12972x2315-*.npy")) + sorted(cache.glob("lookahead-*.pkl"))
    if not files:
        raise SystemExit(f"no caches in {cache}; run a lookahead eval first")
    _upload_dataset(CACHE_DATASET, files, "five-letters-cache", "update caches")


def cmd_push(a) -> None:
    queues = [a.gpu0 or [], a.gpu1 or []]
    if not any(queues):
        raise SystemExit("give at least one --gpu0 or --gpu1 command")
    settings = {"repo": REPO, "branch": a.branch, "subdir": SUBDIR, "queues": queues, "budget_hours": a.hours,
                "pip": a.pip or [], "job": a.name, "results_repo": a.results_repo}
    src = (HERE / "remote_run.py").read_text()
    marker = "SETTINGS = {}  # filled in by kaggle/job.py"
    assert marker in src
    src = src.replace(marker, f"SETTINGS = {json.dumps(settings)}")
    with tempfile.TemporaryDirectory() as d:
        Path(d, "run.py").write_text(src)
        meta = {
            "id": f"{OWNER}/{slug(a.name)}", "title": slug(a.name), "code_file": "run.py",
            "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": True, "enable_internet": True,
            "dataset_sources": [SECRETS_DATASET] + ([CACHE_DATASET] if not a.no_cache else []),
            "competition_sources": [], "kernel_sources": [f"{OWNER}/{slug(k)}" for k in (a.from_kernel or [])],
            "model_sources": [],
        }
        Path(d, "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
        r = kaggle("kernels", "push", "-p", d, "--accelerator", a.accelerator)
    out = (r.stdout + r.stderr).strip()
    print(out)
    if "error" in out.lower():  # the CLI exits 0 even when Kaggle refuses the push
        raise SystemExit(2)
    print(f"queued on {a.accelerator}: https://www.kaggle.com/code/{OWNER}/{slug(a.name)}")
    for g, q in enumerate(queues):
        for c in q:
            print(f"  gpu{g}: {c}")


def cmd_plan(a) -> None:
    """Push one session from a JSON plan file: {"<session>": {"name", "hours", "pip", "gpu0": [...], "gpu1": [...]}}."""
    plans = json.loads(Path(a.file).read_text())
    if a.session not in plans:
        raise SystemExit(f"no session {a.session!r} in {a.file}; choose from {sorted(plans)}")
    p = plans[a.session]
    for ph in ("FINAL_RUN", "BASE_RUN"):
        if ph in json.dumps(p):
            raise SystemExit(f"{a.session} still contains the {ph} placeholder: set it to the run chosen on the select split")
    args = argparse.Namespace(name=p["name"], gpu0=p.get("gpu0", []), gpu1=p.get("gpu1", []),
                              branch=p.get("branch", "development"), hours=p.get("hours", 8.5),
                              accelerator=p.get("accelerator", "NvidiaTeslaT4"), no_cache=False,
                              pip=p.get("pip", []), results_repo=p.get("results_repo", "aafimalek2032/five-letters-results"),
                              from_kernel=p.get("from_kernel", []))
    cmd_push(args)


def cmd_status(a) -> None:
    print(kaggle("kernels", "status", f"{OWNER}/{slug(a.name)}").stdout.strip())


def cmd_fetch(a) -> None:
    dest = Path(a.out or HERE.parent / "kaggle_runs" / a.name)
    dest.mkdir(parents=True, exist_ok=True)
    args = ["kernels", "output", f"{OWNER}/{slug(a.name)}", "-p", str(dest), "-o", "-q"]
    if not a.all:
        args += ["--file-pattern", FETCH_PATTERN]
    kaggle(*args)
    summary = dest / "job_summary.json"
    print(summary.read_text() if summary.exists() else f"no job_summary.json yet (job still running?). Files in {dest}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("secrets", help="upload tokens to a private Kaggle dataset")
    sub.add_parser("cache", help="upload solver caches to a private Kaggle dataset")
    s = sub.add_parser("push", help="start a GPU job")
    s.add_argument("name")
    s.add_argument("--gpu0", action="append", help="a fiveletters command (without 'fiveletters'); repeat for a queue")
    s.add_argument("--gpu1", action="append")
    s.add_argument("--branch", default="development")
    s.add_argument("--hours", type=float, default=8.0, help="session time budget shared by all commands")
    s.add_argument("--accelerator", default="NvidiaTeslaT4", help="NvidiaTeslaT4 = T4 x2")
    s.add_argument("--no-cache", action="store_true")
    s.add_argument("--from-kernel", action="append", help="attach an earlier job's output (its models/) as input")
    s.add_argument("--pip", action="append", help="extra pip packages, e.g. --pip textarena --pip nltk")
    s.add_argument("--results-repo", default="aafimalek2032/five-letters-results",
                   help="private HF dataset that receives results after every command ('' to disable)")
    pl = sub.add_parser("plan", help="push a session defined in a JSON plan file")
    pl.add_argument("file")
    pl.add_argument("session")
    for name in ("status", "fetch"):
        t = sub.add_parser(name)
        t.add_argument("name")
        if name == "fetch":
            t.add_argument("--out", default=None)
            t.add_argument("--all", action="store_true", help="also download checkpoints (large)")
    a = p.parse_args(argv)
    {"secrets": cmd_secrets, "cache": cmd_cache, "push": cmd_push, "plan": cmd_plan, "status": cmd_status,
     "fetch": cmd_fetch}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
