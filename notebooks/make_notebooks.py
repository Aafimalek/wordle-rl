"""Regenerate the Kaggle and Colab launcher notebooks: python notebooks/make_notebooks.py

The code lives in the private repo Aafimalek/rl-environments-startup-guide, under
projects/five-letters. Cloning it needs a read-only GitHub token (secret GITHUB_TOKEN).
The token is sent as an HTTP header for each git command, so it is never written into
the clone's .git/config, never printed, and never saved in Kaggle's notebook output.
"""

import json
from pathlib import Path

HERE = Path(__file__).parent

SETTINGS = '''
REPO = "https://github.com/Aafimalek/rl-environments-startup-guide.git"   # private: needs GITHUB_TOKEN
BRANCH = "development"
SUBDIR = "projects/five-letters"
'''

SECRET_NAMES = '["GITHUB_TOKEN", "HF_TOKEN", "GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "CEREBRAS_API_KEY"]'

INSTALL = r'''
import base64, os, subprocess, sys, importlib.metadata as md
CLONE = os.path.join(BASE, "rl-guide")
WORK = os.path.join(CLONE, SUBDIR)

def git(*args):
    """Run git with the token as a header. On failure, show git's message but never the command (it holds the token)."""
    auth = []
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        auth = ["-c", f"http.extraHeader=Authorization: Basic {basic}"]
    r = subprocess.run(["git", *auth, *args], capture_output=True, text=True)
    if r.returncode != 0:
        msg = (r.stderr or r.stdout).replace(token, "***") if token else (r.stderr or r.stdout)
        raise SystemExit(f"git {args[0]} failed: {msg.strip()}\nIs GITHUB_TOKEN set, and does it have read access to the repo?")
    return r.stdout

if os.path.isdir(os.path.join(CLONE, ".git")):
    git("-C", CLONE, "fetch", "--depth", "1", "origin", BRANCH)
    git("-C", CLONE, "reset", "--hard", "FETCH_HEAD")
else:
    git("clone", "--depth", "1", "-b", BRANCH, REPO, CLONE)
print("code at", git("-C", CLONE, "log", "-1", "--oneline").strip())

# The image already has torch. Install our package without touching it,
# and upgrade transformers only if it is too old for Gemma 3.
subprocess.run(f"pip install -q --no-deps -e {WORK}", shell=True, check=True)
def ver(p):
    try:
        return tuple(int(x) for x in md.version(p).split(".")[:2])
    except Exception:
        return (0, 0)
if ver("transformers") < (4, 53) or ver("huggingface_hub") < (0, 25):
    subprocess.run("pip install -q -U 'transformers>=4.53' 'huggingface_hub>=0.25' safetensors", shell=True, check=True)
subprocess.run("pip install -q hf_xet", shell=True)  # faster, deduplicated Hub uploads
os.chdir(WORK)
'''

RUN = r'''
# Exit code 75 means "paused": the run saved a checkpoint (and uploaded it if hub_repo is set).
# Start a new session and run this notebook again to continue from where it stopped.
import subprocess, sys
subprocess.run([sys.executable, "-m", "fiveletters", "info"])
rc = subprocess.run(f"{sys.executable} -m fiveletters {COMMAND}", shell=True).returncode
print({0: "finished", 75: "paused: rerun this notebook to resume"}.get(rc, f"failed with exit code {rc}: read the log above"))
'''

COMMANDS = '''
# What to run. Edit COMMAND for each job. Examples:
#   "sft --config configs/sft_gemma270m.toml --set checkpoint.hub_repo=YOUR_HF_NAME/five-letters-runs --set data.hub_dataset=YOUR_HF_NAME/five-letters-traces"
#   "grpo --config configs/grpo_gemma270m.toml --set checkpoint.hub_repo=YOUR_HF_NAME/five-letters-runs --set model.init=hub:YOUR_HF_NAME/five-letters-runs:sft-gemma270m-v1/best"
#   "eval --policy hf --model hub:YOUR_HF_NAME/five-letters-runs:sft-gemma270m-v1/best --split heldout"
#   "eval --policy hf --model google/gemma-3-270m-it --split heldout"      (zero-shot baseline)
COMMAND = "info"
'''

TOKEN_HELP = (
    "A **GitHub fine-grained token** with read-only *Contents* access to `rl-environments-startup-guide` only "
    "(GitHub → Settings → Developer settings → Fine-grained tokens). Save it as the secret `GITHUB_TOKEN`."
)


def nb(cells):
    out = []
    for kind, src in cells:
        c = {"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        out.append(c)
    return {"cells": out, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}}}


kaggle = nb([
    ("markdown", f'''
# Five Letters: Kaggle launcher

**Before the first run**
1. Settings (right panel): *Accelerator* **GPU T4 x2**, *Internet* **on**. Both need a verified phone.
2. *Add-ons → Secrets*, add:
   * `GITHUB_TOKEN`: {TOKEN_HELP}
   * `HF_TOKEN`: a Hugging Face **write** token.
   * Optional: `GEMINI_API_KEY`, `GROQ_API_KEY`, `OPENROUTER_API_KEY`.
3. Accept the Gemma 3 license on https://huggingface.co/google/gemma-3-270m-it with the same HF account.
4. Keep this notebook **private** (the Kaggle default).

**Running without keeping the tab open:** *Save Version → Save & Run All (Commit)*. It runs headless until done or the session limit. The run stops itself before Kaggle's limit (time budget), saves, and uploads to your HF runs repo.

**Quota:** about 30 GPU hours a week, reset weekly. CPU jobs (traces, solver and API baselines) use no GPU quota: set the accelerator to *None* for them.
'''),
    ("code", "# 1. Settings\n" + SETTINGS + COMMANDS + 'BASE = "/kaggle/working"\n'),
    ("code", f'''
# 2. Secrets -> environment variables (values are never printed)
import os
try:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    for name in {SECRET_NAMES}:
        try:
            os.environ[name] = secrets.get_secret(name)
        except Exception:
            pass
except ImportError:
    print("Not running on Kaggle")
print("GITHUB_TOKEN set:", bool(os.environ.get("GITHUB_TOKEN")), "| HF_TOKEN set:", bool(os.environ.get("HF_TOKEN")))
'''),
    ("code", "# 3. Get the code (fresh clone, or update an existing one)\n" + INSTALL),
    ("code", "# 4. Run\n" + RUN),
])

colab = nb([
    ("markdown", f'''
# Five Letters: Colab launcher

1. *Runtime → Change runtime type → T4 GPU*.
2. *Secrets* (key icon, left bar), with notebook access switched on:
   * `GITHUB_TOKEN`: {TOKEN_HELP}
   * `HF_TOKEN`: a Hugging Face **write** token. Optional API keys as on Kaggle.
3. Accept the Gemma 3 license on https://huggingface.co/google/gemma-3-270m-it.

Free Colab can disconnect at any time, so keep `checkpoint.hub_repo` set.
Optionally mount Google Drive (free 15 GB) as a second copy: set `USE_DRIVE = True`, and use `--set checkpoint.keep=1` to save space.
'''),
    ("code", SETTINGS + COMMANDS + 'USE_DRIVE = False\nBASE = "/content"\n'),
    ("code", f'''
import os
try:
    from google.colab import userdata
    for name in {SECRET_NAMES}:
        try:
            v = userdata.get(name)
            if v:
                os.environ[name] = v
        except Exception:
            pass
except ImportError:
    print("Not running on Colab")
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    os.environ["FL_RUNS_DIR"] = "/content/drive/MyDrive/five-letters/runs"
print("GITHUB_TOKEN set:", bool(os.environ.get("GITHUB_TOKEN")), "| HF_TOKEN set:", bool(os.environ.get("HF_TOKEN")),
      "| runs dir:", os.environ.get("FL_RUNS_DIR", "default"))
'''),
    ("code", INSTALL),
    ("code", RUN),
])

if __name__ == "__main__":
    for name, obj in (("kaggle_launcher.ipynb", kaggle), ("colab_launcher.ipynb", colab)):
        with open(HERE / name, "w", newline="\n", encoding="utf-8") as f:
            json.dump(obj, f, indent=1)
            f.write("\n")
        print("wrote", HERE / name)
