"""Running on free machines: platform detection, time budgets, crash-safe checkpoints, Hub sync.

How a run survives a free session
---------------------------------
1. A time budget (8 h on Kaggle by default) stops training cleanly before the
   platform kills the session, then saves a checkpoint.
2. Local checkpoints are saved every N minutes into a temp folder that is renamed
   only when complete. A pointer file names the newest one. A killed save never
   corrupts the previous checkpoint.
3. Every M minutes the newest checkpoint is uploaded to a private Hugging Face repo,
   replacing the previous upload. Optionally the repo history is squashed so old
   uploads do not eat free storage.
4. On start, the run resumes from the local checkpoint if there is one, otherwise
   from the Hub copy. A config change between sessions is refused unless forced,
   because silently resuming with different settings ruins an experiment.
5. SIGTERM (sent by some platforms before shutdown) triggers save-and-exit.
6. A finished run writes COMPLETED; rerunning it just reports that and exits.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import signal
import sys
import time
from pathlib import Path

PAUSED_EXIT_CODE = 75  # "try again later": the run saved itself and should be rerun to continue
GATE_FAILED_EXIT_CODE = 76  # a measured result did not justify the next step: stop this queue

DEFAULT_BUDGET_HOURS = {"kaggle": 8.0, "colab": 11.0, "lightning": None, "local": None}


def detect_platform() -> str:
    env = os.environ
    if env.get("KAGGLE_KERNEL_RUN_TYPE") or env.get("KAGGLE_URL_BASE"):
        return "kaggle"
    if "COLAB_RELEASE_TAG" in env or "COLAB_GPU" in env or "google.colab" in sys.modules:
        return "colab"
    if env.get("LIGHTNING_CLOUD_PROJECT_ID") or env.get("LIGHTNING_CLOUDSPACE_HOST"):
        return "lightning"
    return "local"


def default_runs_dir() -> Path:
    if os.environ.get("FL_RUNS_DIR"):
        return Path(os.environ["FL_RUNS_DIR"])
    if detect_platform() == "kaggle":
        return Path("/kaggle/working/runs")
    return Path.cwd() / "runs"


class TimeBudget:
    """Wall-clock budget for this session. hours=None means unlimited. Env FL_TIME_BUDGET_HOURS overrides."""

    def __init__(self, hours: float | str | None = "auto", reserve_minutes: float = 20.0):
        env = os.environ.get("FL_TIME_BUDGET_HOURS")
        if env:
            hours = float(env)
        elif hours == "auto":
            hours = DEFAULT_BUDGET_HOURS.get(detect_platform())
        self.hours = None if hours in (None, 0, "none") else float(hours)
        self.reserve_s = reserve_minutes * 60
        self.start = time.monotonic()
        # Several commands in one session (kaggle/remote_run.py) share one budget: count from session start.
        session_start = os.environ.get("FL_SESSION_START")
        self.offset_s = max(0.0, time.time() - float(session_start)) if session_start else 0.0

    def elapsed_s(self) -> float:
        return time.monotonic() - self.start + self.offset_s

    def remaining_s(self) -> float:
        if self.hours is None:
            return float("inf")
        return self.hours * 3600 - self.elapsed_s()

    def should_stop(self) -> bool:
        return self.remaining_s() <= self.reserve_s

    def describe(self) -> str:
        return "unlimited" if self.hours is None else f"{self.hours:g} h (stops {self.reserve_s / 60:.0f} min early to save)"


class StopFlag:
    """Set by SIGTERM/SIGINT so the training loop can save before exiting."""

    def __init__(self):
        self.reason: str | None = None
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, self._handler)
            except (ValueError, OSError):  # not on the main thread, or unsupported on this OS
                pass

    def _handler(self, signum, _frame):
        if self.reason is not None:  # second signal: give up immediately
            raise KeyboardInterrupt
        self.reason = signal.Signals(signum).name
        print(f"[run] received {self.reason}; saving a checkpoint and stopping", flush=True)

    def __bool__(self):
        return self.reason is not None


def provenance() -> dict:
    """Everything needed to reproduce a number: code commit, library versions, hardware, platform."""
    import importlib.metadata as md
    import platform
    import subprocess

    def ver(pkg):
        try:
            return md.version(pkg)
        except md.PackageNotFoundError:
            return None

    commit = os.environ.get("FL_GIT_COMMIT")
    dirty = None
    if not commit:
        here = Path(__file__).resolve().parent
        try:
            commit = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"], capture_output=True, text=True,
                                    timeout=10).stdout.strip() or None
            dirty = bool(subprocess.run(["git", "-C", str(here), "status", "--porcelain", "--", "."],
                                        capture_output=True, text=True, timeout=10).stdout.strip())
        except (OSError, subprocess.SubprocessError):
            pass
    info = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": detect_platform(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "git_commit": commit,
        "git_dirty": dirty,
        "packages": {p: ver(p) for p in ("fiveletters", "torch", "transformers", "tokenizers", "huggingface_hub",
                                         "safetensors", "numpy", "accelerate")},
    }
    try:
        import torch

        info["cuda"] = torch.version.cuda
        info["gpus"] = [{"name": torch.cuda.get_device_name(i),
                         "capability": ".".join(map(str, torch.cuda.get_device_capability(i))),
                         "memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1)}
                        for i in range(torch.cuda.device_count())]
    except ImportError:
        pass
    return info


def record_provenance(directory: Path, **extra) -> dict:
    """Append this session's provenance to <directory>/provenance.jsonl (one line per session)."""
    directory.mkdir(parents=True, exist_ok=True)
    rec = {**provenance(), **extra}
    with open(directory / "provenance.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def rng_state() -> dict:
    import numpy as np
    import torch

    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # torch.load(map_location="cuda") moves these to the GPU, but set_rng_state needs CPU ByteTensors
    torch.set_rng_state(state["torch"].cpu().to(torch.uint8))
    if "cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in state["cuda"]])
        except RuntimeError:
            pass  # different GPU count than last session (e.g. T4 x2 -> x1)


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class HubSync:
    """Mirror a run's newest checkpoint to a private HF model repo. Failures never stop training."""

    def __init__(self, repo_id: str, run_name: str, *, private: bool = True, squash_history: bool = True,
                 include_optimizer: bool = True):
        self.repo_id, self.run_name = repo_id, run_name
        self.private, self.squash_history, self.include_optimizer = private, squash_history, include_optimizer
        self._api = None

    @property
    def api(self):
        if self._api is None:
            from huggingface_hub import HfApi

            self._api = HfApi()
            self._api.create_repo(self.repo_id, private=self.private, exist_ok=True, repo_type="model")
        return self._api

    def upload(self, local_dir: Path, slot: str = "latest", attempts: int = 3) -> bool:
        ignore = None if self.include_optimizer else ["trainer_state.pt"]
        for attempt in range(1, attempts + 1):
            try:
                self.api.upload_folder(
                    repo_id=self.repo_id, folder_path=str(local_dir), path_in_repo=f"{self.run_name}/{slot}",
                    commit_message=f"{self.run_name}: {slot} <- {local_dir.name}", ignore_patterns=ignore,
                    delete_patterns=["*"],  # remove files of the previous upload in this slot
                )
                if self.squash_history:
                    try:
                        self.api.super_squash_history(repo_id=self.repo_id)
                    except Exception as e:  # noqa: BLE001 - storage cleanup is best effort
                        print(f"[hub] history squash skipped: {e}")
                print(f"[hub] uploaded {local_dir.name} -> {self.repo_id}/{self.run_name}/{slot}")
                return True
            except Exception as e:  # noqa: BLE001 - network or quota trouble must not kill a run
                r = getattr(e, "response", None)
                detail = f"HTTP {r.status_code}: {r.text[:300]}" if r is not None else str(e)[:300]
                print(f"[hub] upload attempt {attempt}/{attempts} of {slot} failed: {type(e).__name__} {detail}",
                      flush=True)
                if attempt < attempts:
                    time.sleep(20 * attempt)
        print("[hub] giving up on this sync; models are still kept locally and exported to the Kaggle output")
        return False

    def delete_slot(self, slot: str = "latest") -> None:
        """Remove a slot (e.g. the resume checkpoint of a finished run) to keep private storage small."""
        try:
            self.api.delete_folder(path_in_repo=f"{self.run_name}/{slot}", repo_id=self.repo_id,
                                   commit_message=f"{self.run_name}: run finished, drop {slot}")
            if self.squash_history:
                self.api.super_squash_history(repo_id=self.repo_id)
            print(f"[hub] removed {self.run_name}/{slot} (run finished; best/final kept)")
        except Exception as e:  # noqa: BLE001
            print(f"[hub] could not remove {slot}: {e}")

    def download(self, dest: Path, slot: str = "latest") -> Path | None:
        try:
            from huggingface_hub import snapshot_download

            path = snapshot_download(self.repo_id, allow_patterns=[f"{self.run_name}/{slot}/*"], local_dir=str(dest))
        except Exception as e:  # noqa: BLE001
            print(f"[hub] no checkpoint downloaded ({e})")
            return None
        p = Path(path) / self.run_name / slot
        return p if (p / "trainer_state.pt").exists() else None


def heavy_dir_for(run_dir: Path) -> Path:
    """Where large files (checkpoints, best/final models, Hub downloads) go. FL_HEAVY_DIR points this at a
    session's scratch disk, so the 20 GB Kaggle output folder only holds logs, metrics and results."""
    root = os.environ.get("FL_HEAVY_DIR")
    return Path(root) / run_dir.name if root else run_dir


def _dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


def is_disk_full(e: BaseException) -> bool:
    import errno

    return (isinstance(e, OSError) and e.errno == errno.ENOSPC) or "No space left" in str(e)         or "unexpected pos" in str(e)  # torch.save's message when the disk fills mid-write


class Checkpointer:
    def __init__(self, run_dir: Path, *, keep: int = 2, every_minutes: float = 20.0, hub: HubSync | None = None,
                 hub_every_minutes: float = 30.0):
        self.run_dir = run_dir
        self.heavy = heavy_dir_for(run_dir)
        self.ckpt_dir = self.heavy / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.keep, self.every_s = keep, every_minutes * 60
        self.hub, self.hub_every_s = hub, hub_every_minutes * 60
        self._last_save = time.monotonic()
        self._last_hub = time.monotonic()
        self._dirty_slots: set[str] = set()

    def mark_dirty(self, slot: str) -> None:
        """run_dir/<slot> (e.g. 'best') changed; upload it with the next Hub sync."""
        self._dirty_slots.add(slot)

    def due(self) -> bool:
        return time.monotonic() - self._last_save >= self.every_s

    def latest(self) -> Path | None:
        ptr = self.ckpt_dir / "LATEST"
        if ptr.exists():
            p = self.ckpt_dir / ptr.read_text().strip()
            if (p / "trainer_state.pt").exists():
                return p
        done = sorted(p for p in self.ckpt_dir.glob("step-*") if not p.name.endswith(".tmp") and (p / "trainer_state.pt").exists())
        return done[-1] if done else None

    def find_resume_point(self) -> Path | None:
        local = self.latest()
        if local:
            print(f"[run] resuming from local checkpoint {local}")
            return local
        if self.hub:
            p = self.hub.download(self.heavy / "hub-download")
            if p:
                print(f"[run] resuming from Hub checkpoint {self.hub.repo_id}/{self.hub.run_name}/latest")
                return p
        return None

    def _make_room(self) -> bool:
        """Free space for one more checkpoint by dropping older ones first. False if it still will not fit."""
        latest = self.latest()
        need = int(_dir_size(latest) * 1.15) + (256 << 20) if latest else 0
        if shutil.disk_usage(self.ckpt_dir).free >= need:
            return True
        done = sorted(p for p in self.ckpt_dir.glob("step-*") if p.is_dir() and not p.name.endswith(".tmp"))
        for p in done[:-1]:  # keep only the newest while writing the next one
            shutil.rmtree(p, ignore_errors=True)
        if shutil.disk_usage(self.ckpt_dir).free >= need:
            return True
        if latest and self.hub:  # last resort: the Hub copy is the backup, the local one goes
            shutil.rmtree(latest, ignore_errors=True)
            return shutil.disk_usage(self.ckpt_dir).free >= need // 2
        return False

    def save(self, step: int, write_fn, *, force_hub: bool = False) -> Path | None:
        """write_fn(tmp_dir) writes model, tokenizer and trainer_state.pt into tmp_dir.
        A full disk never stops training: the save is skipped with a warning and the Hub copy stays the backup."""
        final = self.ckpt_dir / f"step-{step:08d}"
        tmp = self.ckpt_dir / f"step-{step:08d}.tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        if not self._make_room():
            print(f"[run] WARNING: not enough disk for a checkpoint at step {step}; skipped (free "
                  f"{shutil.disk_usage(self.ckpt_dir).free / 2**30:.1f} GB)", flush=True)
            self._last_save = time.monotonic()
            return None
        tmp.mkdir(parents=True)
        try:
            write_fn(tmp)
        except Exception as e:
            shutil.rmtree(tmp, ignore_errors=True)
            if not is_disk_full(e):
                raise
            print(f"[run] WARNING: disk filled while saving step {step}; checkpoint skipped, training continues",
                  flush=True)
            self._last_save = time.monotonic()
            return None
        if final.exists():
            shutil.rmtree(final)
        os.replace(tmp, final)
        ptr_tmp = self.ckpt_dir / "LATEST.tmp"
        ptr_tmp.write_text(final.name)
        os.replace(ptr_tmp, self.ckpt_dir / "LATEST")
        self._last_save = time.monotonic()
        self._prune(final)
        print(f"[run] checkpoint saved: {final}")
        if self.hub and (force_hub or time.monotonic() - self._last_hub >= self.hub_every_s):
            if self.hub.upload(final):
                self._last_hub = time.monotonic()
                for slot in sorted(self._dirty_slots):
                    if (self.heavy / slot).exists() and self.hub.upload(self.heavy / slot, slot=slot):
                        self._dirty_slots.discard(slot)
        return final

    def _prune(self, newest: Path) -> None:
        for stale in self.ckpt_dir.glob("step-*.tmp"):
            shutil.rmtree(stale, ignore_errors=True)
        done = sorted(p for p in self.ckpt_dir.glob("step-*") if p.is_dir() and not p.name.endswith(".tmp"))
        for p in done[: -self.keep]:
            if p != newest:
                shutil.rmtree(p, ignore_errors=True)
        hub_dl = self.heavy / "hub-download"
        if hub_dl.exists():
            shutil.rmtree(hub_dl, ignore_errors=True)  # a local checkpoint now exists


class JsonlLogger:
    """Append-only metrics log (always on), plus Trackio when it is installed and enabled."""

    def __init__(self, path: Path, project: str | None = None, run_name: str | None = None, config: dict | None = None):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._trackio = None
        if project:
            try:
                import trackio

                trackio.init(project=project, name=run_name, config=config, resume="allow")
                self._trackio = trackio
            except Exception as e:  # noqa: BLE001 - tracking is optional
                print(f"[log] trackio disabled: {e}")

    def log(self, step: int, **metrics) -> None:
        rec = {"step": step, "time": round(time.time(), 1), **metrics}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        if self._trackio:
            try:
                self._trackio.log(metrics, step=step)
            except Exception:  # noqa: BLE001
                pass
        shown = " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())
        print(f"[step {step}] {shown}", flush=True)


# Sections that describe the session, not the experiment. Changing them between sessions is fine:
# a new time budget on another platform, or adding a Hub repo halfway through a run.
SESSION_SECTIONS = ("budget", "checkpoint")


def experiment_hash(cfg: dict) -> str:
    return config_hash({k: v for k, v in cfg.items() if k not in SESSION_SECTIONS})


def check_or_write_config(run_dir: Path, cfg: dict, allow_change: bool) -> None:
    path = run_dir / "config.json"
    if path.exists():
        old = json.loads(path.read_text())
        if experiment_hash(old) != experiment_hash(cfg):
            if not allow_change:
                raise RuntimeError(
                    f"{run_dir} was started with a different config. Use a new run name, "
                    "or pass --allow-config-change if the change is deliberate (it is recorded)."
                )
            with open(run_dir / "config_changes.jsonl", "a") as f:
                f.write(json.dumps({"time": time.time(), "old": old, "new": cfg}) + "\n")
    run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2, default=str))


def mark_completed(run_dir: Path, info: dict) -> None:
    (run_dir / "COMPLETED").write_text(json.dumps({"time": time.time(), **info}, indent=2))


def is_completed(run_dir: Path) -> dict | None:
    p = run_dir / "COMPLETED"
    return json.loads(p.read_text()) if p.exists() else None
