from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path

import torch

from ..eval.harness import play_games
from ..eval.metrics import summarize
from ..game import Game
from ..policies.hf import HFPolicy
from ..prompts import PromptFormat
from ..runtime import Checkpointer, HubSync, default_runs_dir, rng_state, set_rng_state
from ..words import get_split


def run_dir_for(cfg: dict) -> Path:
    root = Path(cfg["run"]["runs_dir"]) if cfg["run"].get("runs_dir") else default_runs_dir()
    return root / cfg["run"]["name"]


def make_checkpointer(cfg: dict, run_dir: Path) -> Checkpointer:
    ck = cfg["checkpoint"]
    hub = None
    if ck.get("hub_repo"):
        if not os.environ.get("HF_TOKEN") and not _hf_logged_in():
            print("[hub] warning: no HF_TOKEN found; checkpoints stay local only. Add HF_TOKEN as a notebook secret.")
        else:
            hub = HubSync(ck["hub_repo"], cfg["run"]["name"], squash_history=ck["squash_hub_history"],
                          include_optimizer=ck["hub_include_optimizer"])
    else:
        print("[hub] checkpoint.hub_repo is empty: a dead session can only resume from local files. "
              "Set it to survive Kaggle/Colab resets.")
    return Checkpointer(run_dir, keep=ck["keep"], every_minutes=ck["every_minutes"], hub=hub,
                        hub_every_minutes=ck["hub_every_minutes"])


def _hf_logged_in() -> bool:
    try:
        from huggingface_hub import get_token

        return bool(get_token())
    except Exception:  # noqa: BLE001
        return False


def local_model_candidates(sub: str) -> list[Path]:
    """Where a '<run>/<slot>' model may already exist without the Hub: this session's scratch disk, this
    session's exported models, or an earlier session's output attached as a Kaggle input."""
    import glob

    out = []
    if os.environ.get("FL_HEAVY_DIR"):
        out.append(Path(os.environ["FL_HEAVY_DIR"]) / sub)
    if os.environ.get("FL_MODEL_EXPORT_DIR"):
        out.append(Path(os.environ["FL_MODEL_EXPORT_DIR"]) / sub)
    out += [Path(p) for p in glob.glob(f"/kaggle/input/**/models/{sub}", recursive=True)]
    return out


def _is_model_dir(p: Path) -> bool:
    return (p / "config.json").exists() and any(p.glob("*.safetensors"))


def resolve_model_path(spec: str) -> str:
    """A local folder, a Hub model id, or 'hub:<repo_id>:<run>/<slot>' for a checkpoint slot in a runs repo.
    A hub: spec is resolved from local copies first, so a full or failing Hub cannot break an evaluation."""
    if spec.startswith("hub:"):
        _, repo_id, sub = spec.split(":", 2)
        for cand in local_model_candidates(sub):
            if _is_model_dir(cand):
                print(f"[model] using local copy {cand}")
                return str(cand)
        from huggingface_hub import snapshot_download

        dest = default_runs_dir() / "_downloads" / repo_id.replace("/", "__")
        path = Path(snapshot_download(repo_id, allow_patterns=[f"{sub}/*"], local_dir=str(dest))) / sub
        if not _is_model_dir(path):
            raise FileNotFoundError(f"model {sub} is neither local nor on the Hub repo {repo_id}")
        return str(path)
    return spec


def export_model(src: Path, run_name: str, slot: str) -> None:
    """Copy a finished best/final model to FL_MODEL_EXPORT_DIR (Kaggle output, kept when the session ends),
    independent of whether the Hub upload works."""
    root = os.environ.get("FL_MODEL_EXPORT_DIR")
    if not root or not src.exists():
        return
    dest = Path(root) / run_name / slot
    try:
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        print(f"[model] exported {run_name}/{slot} -> {dest}")
    except OSError as e:
        print(f"[model] WARNING: export of {run_name}/{slot} failed: {e}")


def atomic_save_model(model, tok, dest: Path, fmt: PromptFormat | None = None) -> bool:
    """Write a model folder atomically. On a full disk, clean up, warn and return False instead of crashing."""
    from ..runtime import is_disk_full

    tmp = dest.with_name(dest.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    try:
        model.save_pretrained(tmp, safe_serialization=True)
        tok.save_pretrained(tmp)
        if fmt is not None:  # evaluations read it back, so a model is never played in the wrong format
            (tmp / "fiveletters_format.json").write_text(json.dumps(fmt.to_dict()))
    except Exception as e:
        shutil.rmtree(tmp, ignore_errors=True)
        if not is_disk_full(e):
            raise
        print(f"[run] WARNING: disk full; could not save {dest.name}", flush=True)
        return False
    if dest.exists():
        shutil.rmtree(dest)
    os.replace(tmp, dest)
    return True


def save_trainer_state(path: Path, **state) -> None:
    torch.save({**state, "rng": rng_state()}, path / "trainer_state.pt")


def load_trainer_state(ckpt: Path, device: str) -> dict:
    # Our own checkpoint file. It holds RNG state (numpy arrays), so weights_only must be off.
    state = torch.load(ckpt / "trainer_state.pt", map_location=device, weights_only=False)
    state["rng"] = torch.load(ckpt / "trainer_state.pt", map_location="cpu", weights_only=False)["rng"]
    set_rng_state(state.pop("rng"))
    return state


def lr_lambda(warmup: int, total: int):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return f


def make_optimizer(model, lr: float, weight_decay: float):
    params = [p for p in model.parameters() if p.requires_grad]
    fused = torch.cuda.is_available()
    try:
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, fused=fused)
    except (TypeError, RuntimeError):
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def quick_eval(model, tok, split: str, n_games: int, fmt: PromptFormat, batch_size: int,
               hard_mode: bool = False) -> dict:
    words = get_split(split)[:n_games]
    policy = HFPolicy("in-training", fmt, model=model, tokenizer=tok, batch_size=batch_size)
    games = play_games(policy, [Game(w, hard_mode=hard_mode) for w in words])
    m = summarize([g.record() for g in games], n_boot=200)
    return {"win_rate": m["win_rate"], "mean_guesses_fail7": m["mean_guesses_fail7"],
            "invalid_turn_rate": m["invalid_turn_rate"]}


def better(new: dict, best: dict | None) -> bool:
    if best is None:
        return True
    return (new["win_rate"], -new["mean_guesses_fail7"]) > (best["win_rate"], -best["mean_guesses_fail7"])
