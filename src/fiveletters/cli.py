"""Command line: ``python -m fiveletters <command>``. Run ``python -m fiveletters -h`` for the list."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

from . import PROTOCOL_VERSION, __version__


def cmd_info(_args) -> int:
    from .runtime import TimeBudget, default_runs_dir, detect_platform

    print(f"fiveletters {__version__}, protocol {PROTOCOL_VERSION}, Python {platform.python_version()}")
    print(f"platform: {detect_platform()}  runs dir: {default_runs_dir()}  time budget: {TimeBudget().describe()}")
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                print(f"gpu {i}: {p.name}, {p.total_memory / 2**30:.1f} GB, bf16={torch.cuda.is_bf16_supported()}")
            if not torch.cuda.is_bf16_supported():
                print("  no bfloat16 (T4/P100): training runs in float32, which is safe for Gemma 3")
        else:
            print("gpu: none (CPU only). Fine for solvers, traces, tests and API baselines.")
    except ImportError:
        print("torch: not installed (only needed for model training and local model eval)")
    print(f"HF token: {'set' if os.environ.get('HF_TOKEN') else 'not set'}")
    for env in ("GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "CEREBRAS_API_KEY", "NVIDIA_API_KEY"):
        if os.environ.get(env):
            print(f"{env}: set")
    return 0


def cmd_check(_args) -> int:
    from .words import check_resources

    man = check_resources()
    print("word lists and splits match manifest.json:")
    for name, v in man["splits"].items():
        print(f"  {name:8s} {v['count']:5d} words  sha256 {v['sha256'][:16]}")
    return 0


def _make_policy(args):
    from .prompts import PromptFormat

    fmt = PromptFormat(spaced=not args.unspaced, state_line=not args.no_state)
    kind = args.policy
    if kind.startswith("solver:"):
        from .policies.base import SolverPolicy
        from .solvers import make_solver

        pool = "all" if getattr(args, "split", None) == "rare" else "answers"
        name = kind.split(":", 1)[1]
        if pool == "all" and name == "lookahead":
            print("[eval] lookahead over all 12,972 words is too slow; using entropy for the rare split")
            name = "entropy"
        return SolverPolicy(make_solver(name, pool=pool, hard_mode=getattr(args, "hard", False), seed=args.seed,
                                        vocab=args.vocab), fmt)
    if kind == "hf":
        from .policies.hf import HFPolicy
        from .train.common import resolve_model_path

        tk = {"enable_thinking": False} if args.no_thinking else None
        from .prompts import load_format

        path = resolve_model_path(args.model)
        saved = load_format(path)
        if saved is not None:
            if args.unspaced or args.no_state:
                raise SystemExit("this model saved its prompt format; do not override it with --unspaced/--no-state")
            fmt = saved
            print(f"[eval] prompt format from the model: {fmt.to_dict()}")
        return HFPolicy(path, fmt, batch_size=args.batch_size,
                        max_new_tokens=args.max_new_tokens, template_kwargs=tk, precision=args.precision,
                        constrain=args.constrain, constrain_mode=getattr(args, "constrain_mode", "greedy"),
                        beams=getattr(args, "beams", 16))
    if kind == "api":
        from .policies.api import ChatAPIPolicy

        return ChatAPIPolicy(args.provider, args.model, fmt, requests_per_minute=args.rpm,
                             max_tokens=args.max_tokens, base_url=args.base_url, concurrency=args.concurrency,
                             timeout_s=args.timeout, give_up_after_s=max(900.0, 3 * args.timeout))
    raise SystemExit(f"unknown --policy {kind!r}")


def cmd_eval(args) -> int:
    from .eval.harness import EvalIncomplete, run_eval
    from .runtime import PAUSED_EXIT_CODE, TimeBudget, default_runs_dir
    from .words import get_split

    policy = _make_policy(args)
    words = get_split(args.split)
    out = Path(args.out) if args.out else default_runs_dir() / "eval" / args.split / (
        policy.name.replace("/", "_").replace(":", "_") + ("-hard" if args.hard else "")
        + ("-unspaced" if args.unspaced else "") + ("-nostate" if args.no_state else "")
        + ("-nothinking" if args.policy == "hf" and args.no_thinking else "")
        + (f"-{args.precision}" if args.policy == "hf" and args.precision != "fp32" else "")
        + (f"-constrained-{args.constrain}" if args.policy == "hf" and args.constrain else "")
        + ("-word" if args.policy == "hf" and args.constrain and args.constrain_mode == "word" else "")
        + (f"-retry{args.retries}" if args.retries else "") + ("-open" if args.open_vocab else ""))
    budget = TimeBudget()
    try:
        s = run_eval(policy, words, out, word_set=args.split, hard_mode=args.hard, batch_size=args.batch_size,
                     time_budget_s=None if budget.hours is None else budget.remaining_s() - budget.reserve_s,
                     limit=args.limit, retries=args.retries, open_vocab=args.open_vocab)
    except EvalIncomplete as e:
        print(f"[eval] {e}")
        return PAUSED_EXIT_CODE
    keys = ("games", "win_rate", "win_rate_ci95", "mean_guesses_wins", "mean_guesses_fail7", "invalid_turn_rate", "histogram")
    print(json.dumps({k: s[k] for k in keys if k in s}, indent=2))
    print(f"[eval] saved to {out}")
    return 0


def cmd_public_eval(args) -> int:
    from .eval.harness import EvalIncomplete
    from .public_envs import run_public_eval
    from .runtime import PAUSED_EXIT_CODE, TimeBudget, default_runs_dir

    policy = _make_policy(args)
    lo, hi = (int(x) for x in args.seeds.split(":"))
    out = Path(args.out) if args.out else default_runs_dir() / "public" / f"{args.env}-{args.mode}" / (
        policy.name.replace("/", "_").replace(":", "_") + ("-nothinking" if args.policy == "hf" and args.no_thinking else "")
        + (f"-{args.precision}" if args.policy == "hf" and args.precision != "fp32" else "")
        + (f"-constrained-{args.constrain}" if args.policy == "hf" and args.constrain else "")
        + ("-word" if args.policy == "hf" and args.constrain and args.constrain_mode == "word" else ""))
    budget = TimeBudget()
    try:
        s = run_public_eval(policy, out, env_id=args.env, mode=args.mode, seeds=range(lo, hi), batch_size=args.batch_size,
                            time_budget_s=None if budget.hours is None else budget.remaining_s() - budget.reserve_s)
    except EvalIncomplete as e:
        print(f"[public] {e}")
        return PAUSED_EXIT_CODE
    print(json.dumps({k: s[k] for k in ("all", "clean")}, indent=2))
    print(f"[public] saved to {out}")
    return 0


def cmd_crosscheck(args) -> int:
    from .public_envs import crosscheck_feedback

    r = crosscheck_feedback(args.pairs, args.seed)
    print(json.dumps(r, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(r, indent=2))
    return 0 if r["mismatches"] == 0 else 1


def cmd_traces(args) -> int:
    from .traces import TraceConfig, generate_traces

    cfg = TraceConfig(games_per_word=args.games_per_word, seed=args.seed, noise=args.noise,
                      spaced=not args.unspaced, state_line=not args.no_state, hard_mode=args.hard,
                      vocab=args.vocab, opener=args.opener, invalid_prob=args.invalid_prob,
                      answer_pool="all" if args.answers.startswith("extra") else "answers", teacher=args.teacher)
    m = generate_traces(args.out, cfg, workers=args.workers, split=args.answers)
    print(f"[traces] {m['samples']} samples in {m['shards']} shards at {args.out}")
    if args.push_to:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push_to, repo_type="dataset", private=args.private, exist_ok=True)
        import time

        for attempt in range(1, 6):  # flaky home or notebook connections: retry with backoff
            try:
                api.upload_folder(repo_id=args.push_to, repo_type="dataset", folder_path=args.out,
                                  commit_message="traces", ignore_patterns=["*.tmp"])
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 5:
                    raise
                print(f"[traces] upload attempt {attempt} failed ({type(e).__name__}); retrying in {30 * attempt}s")
                time.sleep(30 * attempt)
        print(f"[traces] uploaded to https://huggingface.co/datasets/{args.push_to}")
    return 0


def cmd_train(args, which: str) -> int:
    from .config import load_config

    if which == "sft":
        from .train.sft import DEFAULTS
        from .train.sft import train_sft as fn
    else:
        from .train.grpo import DEFAULTS
        from .train.grpo import train_grpo as fn
    cfg = load_config(DEFAULTS, args.config, args.set)
    return fn(cfg, allow_config_change=args.allow_config_change)


def cmd_gate(args) -> int:
    """Decide (with --check) or wait for (with --wait-minutes) a go/no-go gate between queued steps.

    A decision is written to <runs>/gates/<name>.json so the other GPU queue can wait on it. A missing
    summary (the step before crashed) counts as a failed gate, so no quota is spent on a broken pipeline."""
    import time

    from .runtime import GATE_FAILED_EXIT_CODE, default_runs_dir

    gate_file = default_runs_dir() / "gates" / f"{args.name}.json"
    if args.mark:
        gate_file.parent.mkdir(parents=True, exist_ok=True)
        gate_file.write_text(json.dumps({"name": args.name, "passed": True, "win_rate": None,
                                         "reason": "marked done by the step before", "summary": None}))
        print(f"[gate] {args.name}: marked done")
        return 0
    if args.check:
        summ = Path(args.check)
        if not summ.is_absolute():
            summ = default_runs_dir() / "eval" / summ
        summ = summ / "summary.json" if summ.suffix != ".json" else summ
        if summ.exists():
            s = json.loads(summ.read_text())
            win = s.get("win_rate")
            passed = win is not None and s.get("games", 0) >= args.min_games and win > args.min_win
            reason = f"win_rate {win} over {s.get('games')} games vs threshold > {args.min_win}"
        else:
            passed, win, reason = False, None, f"missing {summ} (the step before failed)"
        gate_file.parent.mkdir(parents=True, exist_ok=True)
        gate_file.write_text(json.dumps({"name": args.name, "passed": passed, "win_rate": win, "reason": reason,
                                         "summary": str(summ)}, indent=2))
        print(f"[gate] {args.name}: {'PASS' if passed else 'FAIL'} ({reason})")
        return 0 if passed else GATE_FAILED_EXIT_CODE
    deadline = time.time() + args.wait_minutes * 60
    while not gate_file.exists():
        if time.time() > deadline:
            print(f"[gate] {args.name}: no decision after {args.wait_minutes} min; stopping this queue")
            return GATE_FAILED_EXIT_CODE
        time.sleep(30)
    d = json.loads(gate_file.read_text())
    print(f"[gate] {args.name}: {'PASS' if d['passed'] else 'FAIL'} ({d['reason']})")
    return 0 if d["passed"] else GATE_FAILED_EXIT_CODE


def cmd_distill_data(args) -> int:
    from .distill import generate
    from .policies.hf import HFPolicy
    from .prompts import DEFAULT_FORMAT, load_format
    from .runtime import PAUSED_EXIT_CODE, TimeBudget
    from .train.common import resolve_model_path

    path = resolve_model_path(args.model)
    policy = HFPolicy(path, load_format(path) or DEFAULT_FORMAT, batch_size=args.batch_size, constrain=args.constrain,
                      constrain_mode="word", beams=args.beams)
    openers = tuple(None if o.strip() == "model" else o.strip().lower() for o in args.openers.split(","))
    budget = TimeBudget()
    res = generate(policy, args.out, openers=openers, shard_size=args.shard_size, part=args.part, parts=args.parts,
                   limit=args.limit,
                   time_budget_s=None if budget.hours is None else budget.remaining_s() - budget.reserve_s)
    print(f"[distill] {res}")
    return PAUSED_EXIT_CODE if res["paused"] else 0


def cmd_drill(args) -> int:
    from .drill import generate_drill, write_manifest
    from .prompts import PromptFormat

    fmt = PromptFormat(spaced=True, state_line=True, guess_word=args.guess_word)
    res = generate_drill(args.out, args.samples, fmt, seed=args.seed)
    print(f"[drill] {res}")
    cfg_path = __import__("pathlib").Path(args.out) / "trace_config.json"
    if not cfg_path.exists():  # a drill-only folder; when added to a trace folder its config is kept
        cfg_path.write_text(json.dumps({"source": "drill", **fmt.to_dict(), **res}, indent=2))
    print(f"[drill] manifest: {write_manifest(args.out)['samples']} samples in {args.out}")
    return 0


def cmd_recovery_data(args) -> int:
    from .policies.hf import HFPolicy
    from .prompts import DEFAULT_FORMAT, load_format
    from .recovery import generate
    from .train.common import resolve_model_path

    path = resolve_model_path(args.model)
    policy = HFPolicy(path, load_format(path) or DEFAULT_FORMAT, batch_size=args.batch_size)  # alone, greedy, strict
    res = generate(policy, args.out, limit=args.limit, part=args.part, parts=args.parts, per_event=args.per_event,
                   repeat_prob=args.repeat_prob, seed=args.seed)
    print(f"[recovery] {json.dumps(res)}")
    return 0


def cmd_build_v7(args) -> int:
    from .drill import build_v7

    replay = []
    for item in args.replay:
        src, _, frac = item.rpartition(":")
        replay.append((src, float(frac)))
    res = build_v7(args.out, args.per_word, replay, seed=args.seed, strict=args.strict, limit_words=args.limit_words,
                   deep=args.deep, invalid_prob=args.invalid_prob, max_invalid=args.max_invalid)
    print(f"[v7] {json.dumps(res)}")
    if args.push_to:
        import time

        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push_to, repo_type="dataset", private=True, exist_ok=True)
        for attempt in range(1, 6):
            try:
                api.upload_folder(repo_id=args.push_to, repo_type="dataset", folder_path=args.out,
                                  commit_message="traces v7", ignore_patterns=["*.tmp"])
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 5:
                    raise
                print(f"[v7] upload attempt {attempt} failed ({type(e).__name__}); retrying in {30 * attempt}s")
                time.sleep(30 * attempt)
        print(f"[v7] uploaded to https://huggingface.co/datasets/{args.push_to} (private)")
    return 0


def cmd_convert_traces(args) -> int:
    from .drill import convert_traces

    print(f"[convert] {convert_traces(args.src, args.dst)} -> {args.dst} (whole-word guesses)")
    return 0


def cmd_report(args) -> int:
    rows = []
    for p in sorted(Path(args.root).glob("**/summary.json")):
        s = json.loads(p.read_text())
        rows.append((s.get("word_set"), s["policy"].get("name"), s.get("hard_mode"), s.get("games"), s.get("win_rate"),
                     s.get("win_rate_ci95"), s.get("mean_guesses_fail7"), s.get("invalid_turn_rate")))
    if not rows:
        print(f"no summary.json files under {args.root}")
        return 1
    print("| split | policy | hard | games | win rate (95% CI) | mean guesses (loss=7) | invalid turns |")
    print("|---|---|---|---|---|---|---|")
    for split, name, hard, n, win, ci, mean, inv in sorted(rows, key=lambda r: (r[0] or "", -(r[4] or 0))):
        print(f"| {split} | {name} | {'yes' if hard else 'no'} | {n} | {win:.3f} ({ci[0]:.3f}-{ci[1]:.3f}) | {mean:.3f} | {inv:.3f} |")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fiveletters", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info", help="show platform, GPU, time budget and which keys are set")
    sub.add_parser("check", help="verify word lists and frozen splits against the manifest")

    e = sub.add_parser("eval", help="play a policy on a frozen split (resumable)")
    e.add_argument("--policy", required=True, help="solver:lookahead | solver:entropy | solver:random | hf | api")
    e.add_argument("--split", default="heldout",
                   choices=["train", "val", "val300", "select", "heldout", "full", "rare", "wordlebench", "jeffers"])
    e.add_argument("--model", help="hf: model id, local folder, or hub:<repo>:<run>/<slot>. api: model name")
    e.add_argument("--provider", default="gemini", help="api: gemini | groq | openrouter | cerebras | nvidia")
    e.add_argument("--base-url", default=None, help="api: custom OpenAI-compatible endpoint")
    e.add_argument("--rpm", type=float, default=10.0, help="api: max requests per minute (stay under the free limit)")
    e.add_argument("--max-tokens", type=int, default=2048, help="api: max output tokens per turn")
    e.add_argument("--concurrency", type=int, default=8, help="api: requests in flight at once")
    e.add_argument("--timeout", type=float, default=300.0, help="api: seconds to wait for one reply")
    e.add_argument("--max-new-tokens", type=int, default=96, help="hf: max new tokens per turn (raise for thinking models)")
    e.add_argument("--no-thinking", action="store_true", help="hf: pass enable_thinking=False to the chat template (Qwen3)")
    e.add_argument("--constrain", default=None, choices=["wordle", "shared"],
                   help="hf: dictionary-constrained decoding inside <guess> (reported as a separate setting)")
    e.add_argument("--constrain-mode", default="greedy", choices=["greedy", "word"],
                   help="greedy: token by token; word: the model's best whole dictionary word (beam search)")
    e.add_argument("--beams", type=int, default=16)
    e.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"],
                   help="hf: weight precision for inference (fp16 lets a 4B baseline fit a T4). Recorded in meta.json.")
    e.add_argument("--hard", action="store_true", help="hard mode")
    e.add_argument("--open-vocab", action="store_true",
                   help="leaderboard rule (labelled, not strict eval-v1): any five letters a-z are scored, with no "
                        "dictionary check, as WordleBench and Jeffers do")
    e.add_argument("--retries", type=int, default=0,
                   help="retry rule (labelled, not strict eval-v1): a rejected non-word does not use a turn, "
                        "up to this many in a row")
    e.add_argument("--unspaced", action="store_true", help="show words as GHOST instead of G H O S T")
    e.add_argument("--no-state", action="store_true", help="do not ask for a <state> line")
    e.add_argument("--batch-size", type=int, default=64)
    e.add_argument("--limit", type=int, default=None, help="only the first N words (smoke tests)")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--vocab", default="wordle", choices=["wordle", "shared"], help="solver guess vocabulary")
    e.add_argument("--out", default=None)

    t = sub.add_parser("traces", help="generate teacher traces for SFT (CPU; resumable)")
    t.add_argument("--out", default="data/traces/v1")
    t.add_argument("--games-per-word", type=int, default=30)
    t.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--noise", type=float, default=0.1)
    t.add_argument("--unspaced", action="store_true")
    t.add_argument("--no-state", action="store_true")
    t.add_argument("--hard", action="store_true")
    t.add_argument("--vocab", default="wordle", choices=["wordle", "shared"],
                   help="shared: only guesses that both Wordle and TextArena accept")
    t.add_argument("--opener", default=None, help="teacher opening word (default: the solver's choice)")
    t.add_argument("--invalid-prob", type=float, default=0.0, help="chance per game of one invalid turn (recovery)")
    t.add_argument("--teacher", default="optimal", choices=["optimal", "consistent", "common-probe"],
                   help="consistent: every teacher guess fits all clues so far (learnable by a small model)")
    t.add_argument("--answers", default="train", choices=["train", "extra", "extra-broad"],
                   help="train: the 1,752 training answers. extra: ~5.7k non-answer shared-vocabulary words "
                        "(no test words), to teach finding any fitting word")
    t.add_argument("--push-to", default=None, help="HF dataset repo id to upload to, e.g. you/five-letters-traces")
    t.add_argument("--private", action="store_true")

    for name in ("sft", "grpo"):
        s = sub.add_parser(name, help=f"{name.upper()} training (resumes automatically)")
        s.add_argument("--config", default=None)
        s.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value")
        s.add_argument("--allow-config-change", action="store_true")

    pe = sub.add_parser("public-eval", help="play in TextArena Wordle-v0 / verifiers wordle (needs textarena)")
    pe.add_argument("--mode", default="adapted", choices=["adapted", "native-verifiers", "native-textarena"])
    pe.add_argument("--env", default="Wordle-v0")
    pe.add_argument("--policy", required=True, help="solver:lookahead | solver:entropy | hf | api")
    pe.add_argument("--model")
    pe.add_argument("--provider", default="nvidia")
    pe.add_argument("--base-url", default=None)
    pe.add_argument("--rpm", type=float, default=30.0)
    pe.add_argument("--max-tokens", type=int, default=4096)
    pe.add_argument("--concurrency", type=int, default=8)
    pe.add_argument("--timeout", type=float, default=300.0)
    pe.add_argument("--max-new-tokens", type=int, default=96)
    pe.add_argument("--no-thinking", action="store_true")
    pe.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    pe.add_argument("--constrain", default=None, choices=["wordle", "shared"])
    pe.add_argument("--constrain-mode", default="greedy", choices=["greedy", "word"])
    pe.add_argument("--beams", type=int, default=16)
    pe.add_argument("--unspaced", action="store_true")
    pe.add_argument("--no-state", action="store_true")
    pe.add_argument("--vocab", default="shared", choices=["wordle", "shared"])
    pe.add_argument("--seeds", default="0:500", help="task seeds lo:hi (seed i = verifiers task i)")
    pe.add_argument("--batch-size", type=int, default=32)
    pe.add_argument("--seed", type=int, default=0)
    pe.add_argument("--out", default=None)

    cc = sub.add_parser("crosscheck-textarena", help="compare our feedback rule with TextArena's")
    cc.add_argument("--pairs", type=int, default=10000)
    cc.add_argument("--seed", type=int, default=0)
    cc.add_argument("--out", default=None)

    g = sub.add_parser("gate", help="go/no-go between queued steps: --check decides, --wait-minutes waits")
    g.add_argument("name")
    g.add_argument("--check", default=None, help="eval folder (relative to <runs>/eval) or summary.json to test")
    g.add_argument("--min-win", type=float, default=0.0, help="pass only if win_rate is strictly above this")
    g.add_argument("--min-games", type=int, default=1)
    g.add_argument("--wait-minutes", type=float, default=0.0, help="without --check: wait this long for a decision")
    g.add_argument("--mark", action="store_true", help="record that the step before finished (a barrier between GPUs)")

    dr = sub.add_parser("drill", help="round v6: spelling-drill samples (boards -> a real word that fits)")
    dr.add_argument("--out", required=True)
    dr.add_argument("--samples", type=int, default=40000)
    dr.add_argument("--seed", type=int, default=1)
    dr.add_argument("--guess-word", action="store_true", help="write guesses as one lowercase word")
    b7 = sub.add_parser("build-v7", help="round v7: balanced spelling drill + replayed teacher games, one folder")
    b7.add_argument("--out", required=True)
    b7.add_argument("--per-word", type=int, default=30, help="drill samples per common word")
    b7.add_argument("--replay", action="append", default=[], help="TRACE_DIR:SHARE, repeatable")
    b7.add_argument("--strict", action="store_true", help="no dev or test answer is ever a drill target (ablation)")
    b7.add_argument("--seed", type=int, default=1)
    b7.add_argument("--deep", action="store_true", help="v8: deeper boards (1-5 guesses, mostly 3-5)")
    b7.add_argument("--invalid-prob", type=float, default=0.15, help="chance of a rejected non-word on a board")
    b7.add_argument("--max-invalid", type=int, default=1, help="at most this many rejected non-words on a board")
    b7.add_argument("--limit-words", type=int, default=None, help="smoke tests only")
    b7.add_argument("--push-to", default="", help="private HF dataset to upload to")
    cv = sub.add_parser("convert-traces", help="copy spaced traces to the whole-word guess format")
    cv.add_argument("--src", required=True)
    cv.add_argument("--dst", required=True)

    rd = sub.add_parser("recovery-data", help="round v10: the model plays training-side words alone; each rejected "
                                              "non-word becomes boards with a real word as the target")
    rd.add_argument("--model", required=True)
    rd.add_argument("--out", required=True)
    rd.add_argument("--limit", type=int, default=4000, help="most common training-side hidden words to play")
    rd.add_argument("--part", type=int, default=0)
    rd.add_argument("--parts", type=int, default=1)
    rd.add_argument("--per-event", type=int, default=4, help="targets per rejected word")
    rd.add_argument("--repeat-prob", type=float, default=0.3, help="share of boards with the word rejected twice")
    rd.add_argument("--seed", type=int, default=1)
    rd.add_argument("--batch-size", type=int, default=100)

    dd = sub.add_parser("distill-data", help="roadmap 2.1: play training answers with word-level decoding, keep won "
                                             "games as SFT traces (resumable)")
    dd.add_argument("--model", required=True)
    dd.add_argument("--out", required=True)
    dd.add_argument("--openers", default="model,slate", help="comma list; 'model' = the model opens itself")
    dd.add_argument("--part", type=int, default=0)
    dd.add_argument("--parts", type=int, default=1)
    dd.add_argument("--shard-size", type=int, default=200)
    dd.add_argument("--limit", type=int, default=None)
    dd.add_argument("--batch-size", type=int, default=128)
    dd.add_argument("--beams", type=int, default=16)
    dd.add_argument("--constrain", default="wordle", choices=["wordle", "shared"])

    r = sub.add_parser("report", help="markdown table of every eval summary")
    r.add_argument("--root", default="runs/eval")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    handlers = {"info": cmd_info, "check": cmd_check, "eval": cmd_eval, "traces": cmd_traces,
                "sft": lambda a: cmd_train(a, "sft"), "grpo": lambda a: cmd_train(a, "grpo"), "report": cmd_report,
                "public-eval": cmd_public_eval, "crosscheck-textarena": cmd_crosscheck, "gate": cmd_gate,
                "distill-data": cmd_distill_data, "drill": cmd_drill, "convert-traces": cmd_convert_traces,
                "build-v7": cmd_build_v7, "recovery-data": cmd_recovery_data}
    sys.exit(handlers[args.cmd](args))


if __name__ == "__main__":
    main()
