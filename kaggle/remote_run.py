"""Runs inside a Kaggle GPU session (pushed by kaggle/job.py). Do not run locally.

1. Reads tokens from the private dataset `five-letters-secrets` (never printed).
2. Clones this repo's branch with the GitHub token sent as a header (never stored).
3. Installs the package without touching Kaggle's torch.
4. Runs one queue of commands per GPU in parallel (T4 x2), logging to /kaggle/working/logs.
5. Writes job_summary.json. Exit code 75 from a command means "paused, rerun to resume".
"""

import base64
import glob
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time

SETTINGS = {}  # filled in by kaggle/job.py

START = time.time()
os.environ["FL_SESSION_START"] = str(START)  # all commands share one session time budget
os.environ["FL_RUNS_DIR"] = "/kaggle/working/runs"
os.environ["FL_CACHE_DIR"] = "/tmp/fl-cache"
os.environ.setdefault("FL_TIME_BUDGET_HOURS", str(SETTINGS.get("budget_hours", 8.0)))
os.environ["PYTHONUNBUFFERED"] = "1"
# Large files (checkpoints, models) go to the scratch disk with the most free space, never to the
# 20 GB output folder, which Kaggle saves and which only holds logs, metrics and eval results.
def _free_gb(path):
    try:
        return shutil.disk_usage(path).free / 2**30
    except OSError:
        return -1.0


_candidates = {p: _free_gb(p) for p in ("/tmp", "/kaggle/tmp", "/root") if os.path.isdir(p)}
_heavy_root = max(_candidates, key=_candidates.get) if _candidates else "/tmp"
os.environ["FL_HEAVY_DIR"] = os.path.join(_heavy_root, "fl-heavy")
os.environ["FL_MODEL_EXPORT_DIR"] = "/kaggle/working/models"  # kept by Kaggle; no Hub needed
print("disk free (GB):", {**{k: round(v, 1) for k, v in _candidates.items()},
                          "/kaggle/working": round(_free_gb("/kaggle/working"), 1)}, "-> heavy files in", os.environ["FL_HEAVY_DIR"], flush=True)
LOGS = "/kaggle/working/logs"
os.makedirs(LOGS, exist_ok=True)
os.makedirs(os.environ["FL_CACHE_DIR"], exist_ok=True)


def find_input(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    return hits[0] if hits else None


def secret(name):
    p = find_input(name)
    return open(p).read().strip() if p else ""


os.environ["HF_TOKEN"] = secret("hf_token")
GH = secret("github_token")
print("secrets found:", {"hf_token": bool(os.environ["HF_TOKEN"]), "github_token": bool(GH)}, flush=True)


def git(*args):
    auth = []
    if GH:
        basic = base64.b64encode(f"x-access-token:{GH}".encode()).decode()
        auth = ["-c", f"http.extraHeader=Authorization: Basic {basic}"]
    r = subprocess.run(["git", *auth, *args], capture_output=True, text=True)
    if r.returncode != 0:
        msg = (r.stderr or r.stdout)
        if GH:
            msg = msg.replace(GH, "***")
        raise SystemExit(f"git {args[0]} failed: {msg.strip()}")
    return r.stdout


CLONE = "/tmp/rl-guide"
git("clone", "--depth", "1", "-b", SETTINGS["branch"], SETTINGS["repo"], CLONE)
WORK = os.path.join(CLONE, SETTINGS["subdir"])
commit = git("-C", CLONE, "log", "-1", "--format=%h %s").strip()
os.environ["FL_GIT_COMMIT"] = git("-C", CLONE, "rev-parse", "HEAD").strip()  # recorded in every run's provenance
print("code:", SETTINGS["branch"], commit, flush=True)

subprocess.run(f"{sys.executable} -m pip install -q --no-deps -e {WORK}", shell=True, check=True)


def ver(pkg):
    import importlib.metadata as md

    try:
        return tuple(int(x) for x in md.version(pkg).split(".")[:2])
    except Exception:
        return (0, 0)


if ver("transformers") < (4, 53) or ver("huggingface_hub") < (0, 25):
    subprocess.run(f"{sys.executable} -m pip install -q -U 'transformers>=4.53' 'huggingface_hub>=0.25' safetensors",
                   shell=True, check=True)
subprocess.run(f"{sys.executable} -m pip install -q hf_xet", shell=True)
if SETTINGS.get("pip"):  # e.g. textarena + nltk for the public-environment evaluations
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *SETTINGS["pip"]], check=True)

# Pre-built solver caches (pattern table, teacher search) skip minutes of CPU work per session.
for pattern in ("patterns-*.npy", "lookahead-*.pkl"):
    for p in glob.glob(f"/kaggle/input/**/{pattern}", recursive=True):
        shutil.copy(p, os.environ["FL_CACHE_DIR"])

subprocess.run([sys.executable, "-m", "fiveletters", "info"], cwd=WORK)
# The exact software environment of this session, for the paper's reproducibility section.
os.makedirs("/kaggle/working/env", exist_ok=True)
with open("/kaggle/working/env/pip-freeze.txt", "w") as f:
    subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=f, text=True)
with open("/kaggle/working/env/nvidia-smi.txt", "w") as f:
    subprocess.run(["nvidia-smi"], stdout=f, stderr=subprocess.STDOUT, text=True)

results = []
lock = threading.Lock()
upload_lock = threading.Lock()
RESULTS_REPO = SETTINGS.get("results_repo")
SMALL_FILES = ["runs/**/results.jsonl", "runs/**/summary*.json", "runs/**/meta.json", "runs/**/provenance.jsonl",
               "runs/**/metrics.jsonl", "runs/**/config*.json*", "runs/**/COMPLETED", "logs/*", "env/*",
               "job_summary.json"]


def sync_results(reason):
    """Copy small result files to a private HF dataset now, so a cancelled or crashed session loses nothing.
    (Kaggle keeps /kaggle/working only when a session ends normally.)"""
    if not RESULTS_REPO or not os.environ.get("HF_TOKEN"):
        return
    with upload_lock:
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(RESULTS_REPO, repo_type="dataset", private=True, exist_ok=True)
            api.upload_folder(repo_id=RESULTS_REPO, repo_type="dataset", folder_path="/kaggle/working",
                              path_in_repo=SETTINGS["job"], allow_patterns=SMALL_FILES,
                              commit_message=f"{SETTINGS['job']}: {reason}")
            print(f"[results] synced to {RESULTS_REPO}/{SETTINGS['job']} ({reason})", flush=True)
        except Exception as e:  # noqa: BLE001 - never let a sync failure stop the jobs
            print(f"[results] sync failed ({reason}): {e}", flush=True)


def run_queue(gpu, commands):
    for i, cmd in enumerate(commands):
        log_path = f"{LOGS}/gpu{gpu}-{i:02d}.log"
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
        t0 = time.time()
        with open(log_path, "w") as log:
            log.write(f"$ fiveletters {cmd}\n")
            log.flush()
            rc = subprocess.run([sys.executable, "-m", "fiveletters", *shlex.split(cmd)], cwd=WORK, env=env,
                                stdout=log, stderr=subprocess.STDOUT).returncode
        with lock:
            results.append({"gpu": gpu, "cmd": cmd, "rc": rc, "minutes": round((time.time() - t0) / 60, 1), "log": log_path})
        sync_results(f"gpu{gpu} command {i} finished rc={rc}")
        if rc == 75:
            break  # paused on the time budget: later commands would have no time either
        if rc == 76:
            break  # a gate failed: the measured result does not justify the rest of this queue


threads = [threading.Thread(target=run_queue, args=(g, cmds)) for g, cmds in enumerate(SETTINGS["queues"]) if cmds]
for t in threads:
    t.start()


def last_line(path):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 2000))
            lines = [x for x in f.read().decode(errors="replace").splitlines() if x.strip()]
        return lines[-1][:220] if lines else ""
    except OSError:
        return ""


last_sync = time.time()
while any(t.is_alive() for t in threads):
    time.sleep(60)
    if time.time() - last_sync > 1800:  # also every 30 min, for partial eval results
        sync_results("periodic")
        last_sync = time.time()
    now = (time.time() - START) / 60
    for p in sorted(glob.glob(f"{LOGS}/*.log")):
        print(f"[{now:5.0f} min] {os.path.basename(p)}: {last_line(p)}", flush=True)

summary = {"commit": commit, "branch": SETTINGS["branch"], "minutes": round((time.time() - START) / 60, 1),
           "results": results}
with open("/kaggle/working/job_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
sync_results("session finished")
for r in results:
    print(f"\n===== {r['log']} (rc={r['rc']}) =====")
    with open(r["log"], errors="replace") as f:
        print("".join(f.readlines()[-40:]))
