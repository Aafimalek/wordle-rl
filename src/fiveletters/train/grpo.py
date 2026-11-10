"""Multi-turn GRPO in our own Wordle engine.

Each step samples ``answers_per_step`` training answers. For each, ``group_size``
games start from the same state (an optional teacher prefix, the curriculum) and
are played out by sampling from the current model. A game's reward is shared by
every turn the model played in it. Advantages are normalized within each group.
Turns from groups where every game got the same reward carry no signal and are
skipped. With one update per batch the run is on-policy, so the PPO ratio is 1 and
the loss is the plain group-baseline policy gradient. There is no KL term (beta = 0),
which matches current GRPO practice.

The answers and prefixes for step t are a pure function of (seed, t), so a resumed
run continues the same schedule.
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import torch

from ..game import Game
from ..lm import (
    MixedPrecision,
    build_batch,
    chat_text,
    choose_precision,
    completion_logprobs,
    generate,
    load_model_and_tokenizer,
    pick_device,
)
from ..prompts import PromptFormat, parse_guess, parse_state, render_completion, render_prompt
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
from ..solvers import make_solver
from ..state import state_line
from ..words import get_split, load
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
from .rewards import game_reward, group_advantages

DEFAULTS = {
    "run": {"name": "grpo-gemma270m", "seed": 1, "runs_dir": "", "trackio_project": ""},
    "model": {"init": "runs/sft-gemma270m/best", "attn_implementation": "", "gradient_checkpointing": False},
    "format": {"spaced": True, "state_line": True, "hard_mode": False},
    "rollout": {"answers_per_step": 8, "group_size": 8, "temperature": 1.0, "top_p": 1.0,
                "max_new_tokens": 96, "gen_batch_size": 64,
                # share of answers drawn from the 1,610 extra (non-test) words, so RL practises unseen answers
                "extra_answer_frac": 0.0,
                # True = benchmark-clean: public leaderboard words are never used as rollout answers
                "exclude_benchmark_answers": False,
                "exclude_val300_answers": False},
    "reward": {"binary": False, "win": 1.0, "speed_bonus": 0.5, "invalid_penalty": 0.2, "state_bonus": 0.0,
               # per model guess that contradicts clues already known (round v5: teach constraint-following)
               "consistency_penalty": 0.0},
    # P(teacher plays the first k moves) for k = 0, 1, 2, 3. Interpolated from start to end over `steps`.
    "curriculum": {"prefix_probs_start": [0.25, 0.25, 0.25, 0.25], "prefix_probs_end": [1.0, 0.0, 0.0, 0.0],
                   "steps": 300, "teacher": "lookahead", "vocab": "shared", "opener": "crate",  # match traces v2
                   # True: curriculum moves fit all clues (match consistent-teacher traces, round v5)
                   "consistent": False},
    "train": {"steps": 600, "lr": 2e-6, "micro_batch_size": 16, "warmup_steps": 10, "weight_decay": 0.0,
              "grad_clip": 1.0, "log_every": 1, "precision": "auto"},
    "eval": {"every_steps": 50, "split": "val", "games": 100, "batch_size": 100},
    "checkpoint": {"every_minutes": 20.0, "keep": 2, "hub_repo": "", "hub_every_minutes": 30.0,
                   "hub_include_optimizer": False, "squash_hub_history": True},
    "budget": {"hours": "auto", "reserve_minutes": 20.0},
}


def prefix_probs(cfg: dict, step: int) -> list[float]:
    c = cfg["curriculum"]
    a = min(1.0, step / max(1, c["steps"]))
    return [(1 - a) * s + a * e for s, e in zip(c["prefix_probs_start"], c["prefix_probs_end"], strict=True)]


def teacher_line(teacher, answer: str, hard_mode: bool) -> list[str]:
    wl = load()
    g = Game(answer, hard_mode=hard_mode)
    while not g.done:
        g.step("", teacher.choose(g.turns), wl.guess_set)
    return [t.guess for t in g.turns]


def _contradiction_rate(games) -> float:
    from ..game import count_clue_contradictions

    bad = checked = 0
    for g in games:
        b, c = count_clue_contradictions(g.turns)
        bad, checked = bad + b, checked + c
    return bad / max(1, checked)


def plan_step(cfg: dict, step: int, train_words: list[str], extra_words: list[str] | None = None) -> list[tuple[str, int]]:
    rng = random.Random(cfg["run"]["seed"] * 1_000_003 + step)
    n = cfg["rollout"]["answers_per_step"]
    n_extra = round(n * cfg["rollout"].get("extra_answer_frac", 0.0)) if extra_words else 0
    answers = rng.sample(train_words, n - n_extra) + (rng.sample(extra_words, n_extra) if n_extra else [])
    probs = prefix_probs(cfg, step)
    return [(a, rng.choices(range(len(probs)), weights=probs)[0]) for a in answers]


def rollout(model, tok, cfg: dict, plan: list[tuple[str, int]], teacher, fmt: PromptFormat,
            extra_teacher=None, extra_set: frozenset = frozenset()):
    """Play group_size games per planned answer. Returns games, per-turn segments and state accuracy."""
    wl = load()
    ro = cfg["rollout"]
    hard = cfg["format"]["hard_mode"]
    games: list[Game] = []
    for answer, k in plan:
        line = teacher_line(extra_teacher if answer in extra_set else teacher, answer, hard)
        k = min(k, len(line) - 1)  # always leave the model at least one move
        for _ in range(ro["group_size"]):
            g = Game(answer, hard_mode=hard)
            for w in line[:k]:
                g.step(render_completion(g, w, fmt), w, wl.guess_set, by_model=False)
            games.append(g)
    segments: list[tuple[int, list[int], list[int]]] = []
    state_hits = [0] * len(games)
    state_total = [0] * len(games)
    active = [i for i, g in enumerate(games) if not g.done]
    while active:
        prompts = [render_prompt(games[i], fmt) for i in active]
        expected = [state_line(games[i].turns) for i in active]
        outs = generate(model, tok, prompts, max_new_tokens=ro["max_new_tokens"], do_sample=True,
                        temperature=ro["temperature"], top_p=ro["top_p"], batch_size=ro["gen_batch_size"])
        for i, out, exp in zip(active, outs, expected, strict=True):
            games[i].step(out.text, parse_guess(out.text), wl.guess_set)
            if out.completion_ids:
                segments.append((i, out.prompt_ids, out.completion_ids))
            if fmt.state_line:
                state_total[i] += 1
                state_hits[i] += int(parse_state(out.text) == exp)
        active = [i for i in active if not games[i].done]
    return games, segments, state_hits, state_total


def train_grpo(cfg: dict, allow_config_change: bool = False) -> int:
    run_dir = run_dir_for(cfg)
    done = is_completed(run_dir)
    if done:
        print(f"[grpo] {run_dir.name} already completed at step {done.get('step')}; final model on the Hub slot 'final' or in {heavy_dir_for(run_dir) / 'final'}")
        return 0
    check_or_write_config(run_dir, cfg, allow_config_change)
    record_provenance(run_dir, job="grpo")
    budget = TimeBudget(cfg["budget"]["hours"], cfg["budget"]["reserve_minutes"])
    stop = StopFlag()
    seed_everything(cfg["run"]["seed"])
    device = pick_device()
    ckpt = make_checkpointer(cfg, run_dir)
    log = JsonlLogger(run_dir / "metrics.jsonl", cfg["run"]["trackio_project"] or None, cfg["run"]["name"], cfg)
    fmt = PromptFormat(spaced=cfg["format"]["spaced"], state_line=cfg["format"]["state_line"])
    print(f"[grpo] run {run_dir} on {device}; time budget {budget.describe()}")

    resume = ckpt.find_resume_point()
    src = str(resume) if resume else resolve_model_path(cfg["model"]["init"])
    model, tok = load_model_and_tokenizer(src, device, for_training=True,
                                          attn_implementation=cfg["model"]["attn_implementation"] or None)
    if cfg["model"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
    tr = cfg["train"]
    total = tr["steps"]
    opt = make_optimizer(model, tr["lr"], tr["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(tr["warmup_steps"], total))
    probe = []
    for w in get_split("train")[:8]:
        g = Game(w, hard_mode=cfg["format"]["hard_mode"])
        g.step("", "salet", load().guess_set)
        probe.append((tok(chat_text(tok, render_prompt(g, fmt)), add_special_tokens=False)["input_ids"],
                      tok(render_completion(g, w, fmt), add_special_tokens=False)["input_ids"]))
    mp = MixedPrecision(choose_precision(model, probe, tok.pad_token_id, tr["precision"]), device)
    step, best, wall = 0, None, 0.0
    if resume:
        st = load_trainer_state(resume, device)
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        mp.load_state_dict(st.get("precision"))
        step, best, wall = st["step"], st.get("best"), st.get("wall_time_s", 0.0)
        print(f"[grpo] resumed at step {step}/{total}")
    session_start = time.monotonic()
    fit = cfg["format"]["hard_mode"] or cfg["curriculum"]["consistent"]  # solver guesses restricted to fitting words
    teacher = make_solver(cfg["curriculum"]["teacher"], hard_mode=fit,
                          vocab=cfg["curriculum"]["vocab"], opener=cfg["curriculum"]["opener"] or None)
    train_words = get_split("train")
    extra_words, extra_teacher = None, None
    if cfg["rollout"]["extra_answer_frac"] > 0:
        from ..solvers import EntropySolver
        from ..words import extra_answer_words

        extra_words = extra_answer_words()
        extra_teacher = EntropySolver(pool="all", hard_mode=fit,
                                      vocab=cfg["curriculum"]["vocab"], opener=cfg["curriculum"]["opener"] or None)
    if cfg["rollout"]["exclude_benchmark_answers"]:
        from ..words import benchmark_words

        bench = benchmark_words()
        train_words = [w for w in train_words if w not in bench]
        extra_words = [w for w in extra_words if w not in bench] if extra_words else extra_words
        print(f"[grpo] benchmark-clean: {len(train_words)} training answers left")
    if cfg["rollout"]["exclude_val300_answers"]:
        v3 = frozenset(get_split("val300"))
        train_words = [w for w in train_words if w not in v3]
        print(f"[grpo] excluding val300: {len(train_words)} training answers left")
    if cfg["eval"]["split"] == "val300" and not cfg["rollout"]["exclude_val300_answers"]:
        raise ValueError("eval.split = val300 needs rollout.exclude_val300_answers = True")
    extra_set = frozenset(extra_words or [])
    G = cfg["rollout"]["group_size"]
    micro = tr["micro_batch_size"]  # halves itself on out-of-memory

    def write_ckpt(tmp: Path):
        model.save_pretrained(tmp, safe_serialization=True)
        tok.save_pretrained(tmp)
        save_trainer_state(tmp, step=step, optimizer=opt.state_dict(), scheduler=sched.state_dict(), best=best,
                           precision=mp.state_dict(), wall_time_s=wall + time.monotonic() - session_start, config_hash=config_hash(cfg))

    def evaluate():
        nonlocal best
        m = quick_eval(model, tok, cfg["eval"]["split"], cfg["eval"]["games"], fmt, cfg["eval"]["batch_size"],
                       hard_mode=cfg["format"]["hard_mode"])
        log.log(step, **{f"{cfg['eval']['split']}/{k}": v for k, v in m.items()})
        if better(m, best):
            best = {**m, "step": step}
            if atomic_save_model(model, tok, ckpt.heavy / "best"):
                ckpt.mark_dirty("best")

    if step == 0 and cfg["eval"]["every_steps"]:
        evaluate()  # the starting point, so every later number has a reference
    while step < total:
        t0 = time.monotonic()
        plan = plan_step(cfg, step, train_words, extra_words)
        games, segments, hits, tots = rollout(model, tok, cfg, plan, teacher, fmt, extra_teacher, extra_set)
        rewards = [game_reward(g, cfg["reward"], h, t) for g, h, t in zip(games, hits, tots, strict=True)]
        adv, flat_frac = group_advantages(rewards, G)
        used = [(adv[i], p, c) for i, p, c in segments if adv[i] != 0.0]
        model.train()
        n_tokens = sum(len(c) for _, _, c in used)
        loss_total = 0.0
        if used:
            used.sort(key=lambda x: len(x[1]) + len(x[2]))
            while True:
                try:
                    for s in range(0, len(used), micro):
                        chunk = used[s : s + micro]
                        ids, attn, pos, mask, K = build_batch([(p, c) for _, p, c in chunk], tok.pad_token_id, device)
                        logp = completion_logprobs(model, ids, attn, pos, K)
                        a = torch.tensor([x[0] for x in chunk], device=logp.device).unsqueeze(1)
                        loss = -(a * logp * mask).sum() / n_tokens
                        mp.backward(loss)
                        loss_total += loss.item()
                    break
                except torch.cuda.OutOfMemoryError:
                    opt.zero_grad(set_to_none=True)
                    ids = attn = pos = mask = logp = loss = a = None
                    torch.cuda.empty_cache()
                    if micro == 1:
                        raise
                    micro, loss_total = micro // 2, 0.0
                    print(f"[grpo] out of GPU memory; micro batch -> {micro}", flush=True)
            grad_norm = mp.step(model, opt, tr["grad_clip"])
            mp.check(loss_total)
        else:
            grad_norm = 0.0  # every group had identical rewards: nothing to learn this step
            opt.step()  # a no-op without gradients; keeps the LR schedule tied to steps
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        model_turns = [t for g in games for t in g.turns if t.by_model]
        if step % tr["log_every"] == 0:
            log.log(step, reward=sum(rewards) / len(rewards), win_rate=sum(g.won for g in games) / len(games),
                    invalid_rate=sum(not t.valid for t in model_turns) / max(1, len(model_turns)),
                    state_acc=sum(hits) / max(1, sum(tots)), no_signal_groups=flat_frac,
                    clue_contradiction_rate=_contradiction_rate(games),
                    updated_turns=len(used), grad_norm=grad_norm, lr=sched.get_last_lr()[0],
                    mean_prefix=sum(k for _, k in plan) / len(plan), step_s=time.monotonic() - t0,
                    precision=mp.name)
        if cfg["eval"]["every_steps"] and step % cfg["eval"]["every_steps"] == 0:
            evaluate()
        if budget.should_stop() or stop:
            ckpt.save(step, write_ckpt, force_hub=True)
            getattr(teacher, "save_memo", lambda: None)()
            print(f"[grpo] paused at step {step}/{total} ({stop.reason or 'time budget'}). Rerun the same command to resume.")
            return PAUSED_EXIT_CODE
        if ckpt.due():
            ckpt.save(step, write_ckpt)

    evaluate()
    ckpt.save(step, write_ckpt, force_hub=True)
    saved_final = atomic_save_model(model, tok, ckpt.heavy / "final")
    export_model(ckpt.heavy / "final", run_dir.name, "final")
    export_model(ckpt.heavy / "best", run_dir.name, "best")
    if ckpt.hub:
        if saved_final:
            saved_final = ckpt.hub.upload(ckpt.heavy / "final", slot="final")
        best_ok = not (ckpt.heavy / "best").exists() or ckpt.hub.upload(ckpt.heavy / "best", slot="best")
        if saved_final and best_ok:
            ckpt.hub.delete_slot("latest")
    getattr(teacher, "save_memo", lambda: None)()
    mark_completed(run_dir, {"step": step, "best": best})
    print(f"[grpo] done. final: {ckpt.heavy / 'final'}, best ({best}): {ckpt.heavy / 'best'}")
    return 0


def main(cfg: dict, allow_config_change: bool = False) -> None:
    sys.exit(train_grpo(cfg, allow_config_change))
