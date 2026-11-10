"""Supervised fine-tuning on teacher traces (imitation of the lookahead solver).

Loss is on completion tokens only. The data order is a pure function of
(seed, epoch, step), so a resumed run sees exactly the batches it would have seen
without the interruption.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import torch

from ..lm import (
    MixedPrecision,
    build_batch,
    chat_text,
    choose_precision,
    completion_logprobs,
    completion_suffix,
    load_model_and_tokenizer,
    pick_device,
)
from ..prompts import PromptFormat
from ..runtime import (
    PAUSED_EXIT_CODE,
    JsonlLogger,
    StopFlag,
    TimeBudget,
    check_or_write_config,
    config_hash,
    heavy_dir_for,
    is_completed,
    mark_completed,
    record_provenance,
    seed_everything,
)
from ..traces import iter_samples
from .common import (
    atomic_save_model,
    better,
    export_model,
    load_trainer_state,
    lr_lambda,
    make_checkpointer,
    make_optimizer,
    quick_eval,
    resolve_model_path,
    run_dir_for,
    save_trainer_state,
)

DEFAULTS = {
    "run": {"name": "sft-gemma270m", "seed": 1, "runs_dir": "", "trackio_project": ""},
    "model": {"name": "google/gemma-3-270m-it", "attn_implementation": "", "gradient_checkpointing": False},
    "data": {"traces_dir": "data/traces/v2", "hub_dataset": "", "max_length": 512, "max_samples": 0,
             # optional second trace set mixed in, e.g. extra answers (traces v3) for generalization
             "extra_traces_dir": "", "extra_hub_dataset": "",
             # keep this share of the extra traces (deterministic per sample). Used for replay in distillation:
             # a slice of the original SFT data keeps the model's earlier calibration while it learns new targets
             "extra_sample_frac": 1.0,
             # True = benchmark-clean: no game whose answer is a public leaderboard word is trained on
             "exclude_benchmark_answers": False,
             # True = val300 words are never training answers (use with eval.split = "val300")
             "exclude_val300_answers": False},
    "train": {"epochs": 2, "micro_batch_size": 16, "grad_accum": 4, "lr": 5e-5, "warmup_steps": 100,
              "weight_decay": 0.0, "grad_clip": 1.0, "max_steps": 0, "log_every": 10,
              "precision": "auto"},  # auto: bf16 if the GPU has it, else fp16 if it checks out, else fp32
    "eval": {"every_steps": 500, "split": "val", "games": 100, "batch_size": 100},
    "checkpoint": {"every_minutes": 20.0, "keep": 2, "hub_repo": "", "hub_every_minutes": 30.0,
                   "hub_include_optimizer": False, "squash_hub_history": True},
    "budget": {"hours": "auto", "reserve_minutes": 20.0},
}


def _fetch_traces(cfg: dict, dir_key: str = "traces_dir", hub_key: str = "hub_dataset") -> Path:
    d = Path(cfg["data"][dir_key])
    if (d / "manifest.json").exists():
        return d
    if cfg["data"][hub_key]:
        from huggingface_hub import snapshot_download

        print(f"[sft] downloading traces from {cfg['data'][hub_key]}")
        snapshot_download(cfg["data"][hub_key], repo_type="dataset", local_dir=str(d))
        if (d / "manifest.json").exists():
            return d
    raise FileNotFoundError(f"no traces in {d}. Run `fiveletters traces --out {d}` or set data.{hub_key}.")


def _keep(sample: dict, frac: float) -> bool:
    """Deterministic subsample: the same samples are kept on every machine and every resume."""
    if frac >= 1.0:
        return True
    import hashlib

    key = f"{sample['answer']}|{sample.get('game')}|{sample.get('turn')}|{sample['completion']}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64 < frac


def _tokenize(tok, trace_dirs, max_length: int, max_samples: int, exclude: frozenset = frozenset(),
              extra_frac: float = 1.0):
    suffix = completion_suffix(tok)
    prompts, comps, lengths = [], [], []
    skipped = 0
    samples = (s for k, d in enumerate(trace_dirs) for s in iter_samples(d)
               if s["answer"] not in exclude and (k == 0 or _keep(s, extra_frac)))
    for i, s in enumerate(samples):
        if max_samples and i >= max_samples:
            break
        prompts.append(chat_text(tok, s["prompt"]))
        comps.append(s["completion"] + suffix)
    p_ids = tok(prompts, add_special_tokens=False)["input_ids"]
    c_ids = tok(comps, add_special_tokens=False)["input_ids"]
    data = []
    for p, c in zip(p_ids, c_ids, strict=True):
        if len(p) + len(c) > max_length:
            skipped += 1
            continue
        data.append((p, c))
        lengths.append(len(p) + len(c))
    return data, lengths, skipped


def _step_plan(n: int, lengths: list[int], eff: int, micro: int, seed: int, epoch: int) -> list[list[list[int]]]:
    idx = list(range(n))
    random.Random(seed * 1_000_003 + epoch).shuffle(idx)
    plan = []
    for s in range(0, n - eff + 1, eff):
        group = sorted(idx[s : s + eff], key=lambda i: lengths[i])  # similar lengths per micro-batch
        plan.append([group[j : j + micro] for j in range(0, eff, micro)])
    return plan


def train_sft(cfg: dict, allow_config_change: bool = False) -> int:
    run_dir = run_dir_for(cfg)
    done = is_completed(run_dir)
    if done:
        print(f"[sft] {run_dir.name} already completed at step {done.get('step')}; final model on the Hub slot 'final' or in {heavy_dir_for(run_dir) / 'final'}")
        return 0
    check_or_write_config(run_dir, cfg, allow_config_change)
    record_provenance(run_dir, job="sft")
    budget = TimeBudget(cfg["budget"]["hours"], cfg["budget"]["reserve_minutes"])
    stop = StopFlag()
    seed_everything(cfg["run"]["seed"])
    device = pick_device()
    ckpt = make_checkpointer(cfg, run_dir)
    log = JsonlLogger(run_dir / "metrics.jsonl", cfg["run"]["trackio_project"] or None, cfg["run"]["name"], cfg)
    print(f"[sft] run {run_dir} on {device}; time budget {budget.describe()}")

    trace_dir = _fetch_traces(cfg)
    tcfg = json.loads((trace_dir / "trace_config.json").read_text())
    trace_dirs = [trace_dir]
    if cfg["data"]["extra_traces_dir"]:
        extra = _fetch_traces(cfg, "extra_traces_dir", "extra_hub_dataset")
        ecfg = json.loads((extra / "trace_config.json").read_text())
        if (ecfg["spaced"], ecfg["state_line"], ecfg.get("guess_word", False)) != (
                tcfg["spaced"], tcfg["state_line"], tcfg.get("guess_word", False)):
            raise ValueError("extra traces use a different prompt format than the main traces")
        trace_dirs.append(extra)
    fmt = PromptFormat(spaced=tcfg["spaced"], state_line=tcfg["state_line"], guess_word=tcfg.get("guess_word", False))

    resume = ckpt.find_resume_point()
    model_src = str(resume) if resume else resolve_model_path(cfg["model"]["name"])
    model, tok = load_model_and_tokenizer(model_src, device, for_training=True,
                                          attn_implementation=cfg["model"]["attn_implementation"] or None)
    if cfg["model"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    exclude = frozenset()
    if cfg["data"]["exclude_benchmark_answers"]:
        from ..words import benchmark_words

        exclude = benchmark_words()
        print(f"[sft] benchmark-clean: excluding games whose answer is one of {len(exclude)} leaderboard words")
    if cfg["data"]["exclude_val300_answers"]:
        from ..words import get_split

        exclude = exclude | frozenset(get_split("val300"))
        print("[sft] excluding the 300 val300 selection words as answers")
    if cfg["eval"]["split"] == "val300" and not cfg["data"]["exclude_val300_answers"]:
        raise ValueError("eval.split = val300 needs data.exclude_val300_answers = True (otherwise selection is on trained answers)")
    data, lengths, skipped = _tokenize(tok, trace_dirs, cfg["data"]["max_length"], cfg["data"]["max_samples"], exclude,
                                       cfg["data"]["extra_sample_frac"])
    tr = cfg["train"]
    eff = tr["micro_batch_size"] * tr["grad_accum"]
    steps_per_epoch = len(data) // eff
    if steps_per_epoch == 0:
        raise ValueError(f"only {len(data)} samples, fewer than one batch of {eff}")
    total = tr["max_steps"] or tr["epochs"] * steps_per_epoch
    print(f"[sft] {len(data)} samples ({skipped} longer than {cfg['data']['max_length']} tokens skipped), "
          f"{steps_per_epoch} steps/epoch, {total} steps total")

    opt = make_optimizer(model, tr["lr"], tr["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(tr["warmup_steps"], total))
    mp = MixedPrecision(choose_precision(model, data[:8], tok.pad_token_id, tr["precision"]), device)
    step, best, wall = 0, None, 0.0
    if resume:
        st = load_trainer_state(resume, device)
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        mp.load_state_dict(st.get("precision"))
        step, best, wall = st["step"], st.get("best"), st.get("wall_time_s", 0.0)
        print(f"[sft] resumed at step {step}/{total} (best so far: {best})")
    print(f"[sft] precision: {mp.name}")
    session_start = time.monotonic()

    def write_ckpt(tmp: Path):
        model.save_pretrained(tmp, safe_serialization=True)
        tok.save_pretrained(tmp)
        save_trainer_state(tmp, step=step, optimizer=opt.state_dict(), scheduler=sched.state_dict(), best=best,
                           precision=mp.state_dict(), wall_time_s=wall + time.monotonic() - session_start, config_hash=config_hash(cfg))

    def evaluate():
        nonlocal best
        m = quick_eval(model, tok, cfg["eval"]["split"], cfg["eval"]["games"], fmt, cfg["eval"]["batch_size"])
        log.log(step, **{f"{cfg['eval']['split']}/{k}": v for k, v in m.items()})
        if better(m, best):
            best = {**m, "step": step}
            if atomic_save_model(model, tok, ckpt.heavy / "best", fmt):
                ckpt.mark_dirty("best")
            print(f"[sft] new best at step {step}: {m}")

    plan_epoch, plan = -1, []
    start_step = step
    micro = tr["micro_batch_size"]  # halves itself on out-of-memory; effective batch and data order stay the same
    model.train()
    t0, tokens_seen = time.monotonic(), 0
    while step < total:
        epoch, within = divmod(step, steps_per_epoch)
        if epoch != plan_epoch:
            plan, plan_epoch = _step_plan(len(data), lengths, eff, micro, cfg["run"]["seed"], epoch), epoch
        micro_batches = plan[within]
        n_tokens = sum(len(data[i][1]) for mb in micro_batches for i in mb)
        loss_sum = 0.0
        try:
            for mb in micro_batches:
                ids, attn, pos, mask, K = build_batch([data[i] for i in mb], tok.pad_token_id, device)
                logp = completion_logprobs(model, ids, attn, pos, K)
                loss = -(logp * mask).sum() / n_tokens
                mp.backward(loss)
                loss_sum += loss.item()
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            ids = attn = pos = mask = logp = loss = None
            torch.cuda.empty_cache()
            if micro == 1:
                raise
            micro //= 2
            plan_epoch = -1  # re-split this epoch's steps into smaller micro-batches
            print(f"[sft] out of GPU memory; micro batch -> {micro}, grad accumulation -> {eff // micro}", flush=True)
            continue
        grad_norm = mp.step(model, opt, tr["grad_clip"])
        mp.check(loss_sum)
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        tokens_seen += n_tokens
        if step % tr["log_every"] == 0 or step == 1:
            dt = time.monotonic() - t0
            log.log(step, loss=loss_sum, lr=sched.get_last_lr()[0], grad_norm=grad_norm, epoch=epoch,
                    precision=mp.name, completion_tokens_per_s=tokens_seen / max(dt, 1e-6),
                    eta_h=(total - step) * dt / max(1, step - start_step) / 3600)
        if cfg["eval"]["every_steps"] and step % cfg["eval"]["every_steps"] == 0:
            evaluate()
        if budget.should_stop() or stop:
            reason = stop.reason or "time budget"
            ckpt.save(step, write_ckpt, force_hub=True)
            print(f"[sft] paused at step {step}/{total} ({reason}). Rerun the same command to resume.")
            return PAUSED_EXIT_CODE
        if ckpt.due():
            ckpt.save(step, write_ckpt)

    evaluate()
    ckpt.save(step, write_ckpt, force_hub=True)
    saved_final = atomic_save_model(model, tok, ckpt.heavy / "final", fmt)
    export_model(ckpt.heavy / "final", run_dir.name, "final")
    export_model(ckpt.heavy / "best", run_dir.name, "best")
    if ckpt.hub:
        if saved_final:
            saved_final = ckpt.hub.upload(ckpt.heavy / "final", slot="final")
        best_ok = not (ckpt.heavy / "best").exists() or ckpt.hub.upload(ckpt.heavy / "best", slot="best")
        if saved_final and best_ok:
            ckpt.hub.delete_slot("latest")
    mark_completed(run_dir, {"step": step, "best": best})
    print(f"[sft] done. final model: {ckpt.heavy / 'final'}, best ({best}): {ckpt.heavy / 'best'}")
    return 0


def main(cfg: dict, allow_config_change: bool = False) -> None:
    sys.exit(train_sft(cfg, allow_config_change))
