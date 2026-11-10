"""End-to-end training on a tiny random model (CPU): runs, pauses on the time budget, resumes."""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from fiveletters import traces as T  # noqa: E402
from fiveletters.config import load_config  # noqa: E402
from fiveletters.lm import build_batch, chat_text, completion_logprobs, load_model_and_tokenizer  # noqa: E402
from fiveletters.runtime import PAUSED_EXIT_CODE  # noqa: E402
from fiveletters.solvers import EntropySolver  # noqa: E402
from fiveletters.train.rewards import game_reward, group_advantages  # noqa: E402
from fiveletters.words import get_split  # noqa: E402


def test_left_padded_logprobs_match_unpadded(tiny_model_dir):
    model, tok = load_model_and_tokenizer(tiny_model_dir, "cpu", for_training=False)
    pairs = []
    for prompt, comp in (("short", "<guess>G H O S T</guess><eos>"), ("a much longer prompt " * 4, "<guess>C R A N E</guess>")):
        p = tok(chat_text(tok, prompt), add_special_tokens=False)["input_ids"]
        c = tok(comp, add_special_tokens=False)["input_ids"]
        pairs.append((p, c))
    ids, attn, pos, mask, K = build_batch(pairs, tok.pad_token_id, "cpu")
    batched = completion_logprobs(model, ids, attn, pos, K)
    for r, (p, c) in enumerate(pairs):
        full = torch.tensor([p + c])
        logits = model(input_ids=full).logits[0, len(p) - 1 : -1]
        want = torch.log_softmax(logits, -1).gather(-1, torch.tensor(c).unsqueeze(-1)).squeeze(-1)
        got = batched[r, K - len(c):]
        assert torch.allclose(got, want, atol=1e-4)
        assert mask[r].sum().item() == len(c)


def test_rewards_and_advantages():
    from fiveletters.game import Game
    from fiveletters.words import load

    vw = load().guess_set
    g1 = Game("ghost")
    g1.step("", "ghost", vw)
    g2 = Game("ghost")
    g2.step("", None, vw)
    cfg = {"win": 1.0, "speed_bonus": 0.5, "invalid_penalty": 0.2, "state_bonus": 0.0}
    assert game_reward(g1, cfg) == pytest.approx(1.5)
    assert game_reward(g2, cfg) == pytest.approx(-0.2)
    assert game_reward(g1, {"binary": True}) == 1.0
    adv, flat = group_advantages([1.0, 0.0, 0.5, 0.5], 2)
    assert adv[0] > 0 > adv[1] and adv[2:] == [0.0, 0.0] and flat == 0.5


@pytest.fixture
def tiny_traces(tmp_path, monkeypatch):
    T._WORKER.clear()
    s = EntropySolver()
    monkeypatch.setattr(T, "_teachers", lambda cfg: (s, s, ["salet"]))
    d = tmp_path / "traces"
    d.mkdir()
    cfg = T.TraceConfig(games_per_word=2)
    with open(d / "shard-00000.jsonl", "w") as f:
        for w in get_split("train")[:12]:
            for gi in range(2):
                for smp in T.play_teacher_game(w, gi, cfg):
                    f.write(json.dumps(smp) + "\n")
    from dataclasses import asdict

    (d / "trace_config.json").write_text(json.dumps({"split": "train", **asdict(cfg)}))
    (d / "manifest.json").write_text("{}")
    yield d
    T._WORKER.clear()


def _sft_cfg(tiny_model_dir, traces, runs, **budget):
    from fiveletters.train.sft import DEFAULTS

    return load_config(DEFAULTS, None, [
        f"model.name={tiny_model_dir}", f"data.traces_dir={traces}", f"run.runs_dir={runs}", "run.name=t-sft",
        "train.micro_batch_size=4", "train.grad_accum=2", "train.max_steps=6", "train.warmup_steps=1",
        "eval.every_steps=3", "eval.games=2", "eval.batch_size=2",
        "checkpoint.every_minutes=0", "checkpoint.keep=2",
        f"budget.hours={budget.get('hours', 'none')}", f"budget.reserve_minutes={budget.get('reserve', 0)}",
    ])


def test_sft_pauses_and_resumes(tiny_model_dir, tiny_traces, tmp_path):
    from fiveletters.train.sft import train_sft

    runs = tmp_path / "runs"
    assert train_sft(_sft_cfg(tiny_model_dir, tiny_traces, runs, hours=1e-9)) == PAUSED_EXIT_CODE
    run = runs / "t-sft"
    assert (run / "checkpoints" / "LATEST").read_text() == "step-00000001"
    assert train_sft(_sft_cfg(tiny_model_dir, tiny_traces, runs)) == 0
    steps = [json.loads(x)["step"] for x in (run / "metrics.jsonl").read_text().splitlines() if '"loss"' in x]
    assert steps[0] == 1 and max(steps) <= 6 and (run / "COMPLETED").exists() and (run / "final").exists()
    kept = sorted(p.name for p in (run / "checkpoints").glob("step-*"))
    assert kept == ["step-00000005", "step-00000006"]
    assert train_sft(_sft_cfg(tiny_model_dir, tiny_traces, runs)) == 0  # completed: no retraining
    changed = _sft_cfg(tiny_model_dir, tiny_traces, runs)
    changed["train"]["lr"] = 1.0
    (run / "COMPLETED").unlink()
    with pytest.raises(RuntimeError, match="different config"):
        train_sft(changed)


def test_grpo_runs_pauses_resumes(tiny_model_dir, tmp_path):
    from fiveletters.train.grpo import DEFAULTS, train_grpo

    def cfg(hours):
        return load_config(DEFAULTS, None, [
            f"model.init={tiny_model_dir}", f"run.runs_dir={tmp_path}", "run.name=t-grpo", "curriculum.teacher=entropy",
            "rollout.answers_per_step=2", "rollout.group_size=2", "rollout.max_new_tokens=12",
            "train.steps=3", "train.micro_batch_size=4", "eval.every_steps=0", "eval.games=2", "eval.batch_size=2",
            "checkpoint.every_minutes=0", f"budget.hours={hours}", "budget.reserve_minutes=0",
        ])

    assert train_grpo(cfg(1e-9)) == PAUSED_EXIT_CODE
    assert train_grpo(cfg("none")) == 0
    run = tmp_path / "t-grpo"
    logged = [json.loads(x) for x in (run / "metrics.jsonl").read_text().splitlines() if '"reward"' in x]
    assert [r["step"] for r in logged] == [1, 2, 3]
    assert (run / "final").exists() and (run / "COMPLETED").exists()


def test_mixed_precision_fallback_and_resume_state():
    from fiveletters.lm import MixedPrecision, amp_dtype

    mp = MixedPrecision(torch.float16, "cpu", patience=2)
    assert mp.name == "fp16" and mp.scaler is not None and amp_dtype("cpu") == torch.float16
    mp.check(float("nan"))
    assert mp.name == "fp16"
    mp.check(float("inf"))
    assert mp.name == "fp32" and mp.fell_back and amp_dtype("cpu") is None
    again = MixedPrecision(torch.float16, "cpu")
    again.load_state_dict(mp.state_dict())  # a later session stays on fp32
    assert again.name == "fp32" and amp_dtype("cpu") is None


def test_choose_precision_on_cpu_is_fp32(tiny_model_dir):
    from fiveletters.lm import choose_precision

    model, tok = load_model_and_tokenizer(tiny_model_dir, "cpu", for_training=True)
    assert choose_precision(model, [([1, 2, 3], [4, 5])], tok.pad_token_id, "auto") is None
    assert choose_precision(model, [], tok.pad_token_id, "fp16") == torch.float16


def test_sft_runs_with_loss_scaling(tiny_model_dir, tiny_traces, tmp_path):
    """Exercises the GradScaler path (used for fp16 on T4) on CPU with bf16 autocast plus a scaler."""
    from fiveletters.lm import set_amp_dtype
    from fiveletters.train.sft import train_sft

    cfg = _sft_cfg(tiny_model_dir, tiny_traces, tmp_path / "runs")
    cfg["run"]["name"] = "t-sft-fp16"
    cfg["train"]["precision"] = "fp16"
    cfg["train"]["max_steps"] = 3
    try:
        assert train_sft(cfg) == 0
    finally:
        set_amp_dtype(None)
    logged = [json.loads(x) for x in (tmp_path / "runs" / "t-sft-fp16" / "metrics.jsonl").read_text().splitlines() if '"loss"' in x]
    assert logged and logged[0]["precision"] in ("fp16", "fp32")


def test_generate_halves_batch_on_oom(tiny_model_dir, monkeypatch):
    from fiveletters import lm

    model, tok = load_model_and_tokenizer(tiny_model_dir, "cpu", for_training=False)
    real = lm._generate_chunk
    sizes = []

    def flaky(model_, tok_, idx, *a, **k):
        sizes.append(len(idx))
        if len(idx) > 2:
            raise torch.cuda.OutOfMemoryError("simulated")
        return real(model_, tok_, idx, *a, **k)

    monkeypatch.setattr(lm, "_generate_chunk", flaky)
    lm._GEN_BATCH_CAP.clear()
    outs = lm.generate(model, tok, ["a", "b", "c", "d", "e"], max_new_tokens=3, batch_size=8)
    assert len(outs) == 5 and all(o is not None for o in outs)
    assert sizes[:3] == [5, 4, 2] or sizes[0] > 2  # 8 -> 4 -> 2, then all chunks of 2 or fewer
    assert max(sizes[2:]) <= 2 and lm._GEN_BATCH_CAP[id(model)] == 2


def test_full_disk_never_stops_training(tmp_path, monkeypatch):
    import errno
    import shutil as sh

    from fiveletters.runtime import Checkpointer

    ck = Checkpointer(tmp_path / "run", keep=2, every_minutes=0)

    def ok(d):
        (d / "trainer_state.pt").write_bytes(b"x" * 1000)

    def full(d):
        (d / "trainer_state.pt").write_bytes(b"x" * 10)
        raise OSError(errno.ENOSPC, "No space left on device")

    assert ck.save(1, ok) is not None
    assert ck.save(2, full) is None  # skipped, not raised
    assert not list(ck.ckpt_dir.glob("*.tmp")) and ck.latest().name == "step-00000001"

    Usage = type(sh.disk_usage(tmp_path))
    monkeypatch.setattr("fiveletters.runtime.shutil.disk_usage", lambda p: Usage(10**9, 10**9, 10))
    assert ck.save(3, ok) is None  # no room even after pruning: skipped with a warning
    assert ck.latest().name == "step-00000001"


def test_heavy_files_go_to_scratch_disk(tmp_path, monkeypatch):
    from fiveletters.runtime import Checkpointer

    monkeypatch.setenv("FL_HEAVY_DIR", str(tmp_path / "scratch"))
    ck = Checkpointer(tmp_path / "out" / "run-a", keep=1, every_minutes=0)
    ck.save(1, lambda d: (d / "trainer_state.pt").write_bytes(b"x"))
    assert (tmp_path / "scratch" / "run-a" / "checkpoints" / "step-00000001").exists()
    assert not (tmp_path / "out" / "run-a" / "checkpoints").exists()


def test_rng_state_restores_from_any_device():
    from fiveletters.runtime import rng_state, set_rng_state

    st = rng_state()
    st["torch"] = st["torch"].to(torch.float32)  # what a device move / dtype change can leave behind
    set_rng_state(st)  # must not raise


def test_checkpoint_policy_names_do_not_collide(tiny_model_dir, tmp_path):
    import shutil as sh

    from fiveletters.policies.hf import HFPolicy

    for run in ("sft-a", "sft-b"):
        sh.copytree(tiny_model_dir, tmp_path / run / "best")
    a = HFPolicy(str(tmp_path / "sft-a" / "best"), device="cpu").name
    b = HFPolicy(str(tmp_path / "sft-b" / "best"), device="cpu").name
    assert a == "hf-sft-a-best" and b == "hf-sft-b-best"


def test_grpo_plan_mixes_extra_answers_deterministically():
    from fiveletters.train.grpo import DEFAULTS, plan_step
    from fiveletters.words import extra_answer_words

    cfg = load_config(DEFAULTS, None, ["rollout.extra_answer_frac=0.5"])
    train, extra = get_split("train"), extra_answer_words()
    p1 = plan_step(cfg, 7, train, extra)
    assert p1 == plan_step(cfg, 7, train, extra)  # same step, same plan: resume-safe
    answers = [a for a, _ in p1]
    assert sum(a in set(extra) for a in answers) == 4 and not set(answers) & set(get_split("heldout"))


def test_benchmark_clean_excludes_leaderboard_answers():
    from fiveletters.train.grpo import DEFAULTS, plan_step
    from fiveletters.words import benchmark_words, extra_answer_words, get_split

    bench = benchmark_words()
    assert len(bench) >= 140 and {"acorn", "apple"} <= bench
    cfg = load_config(DEFAULTS, None, ["rollout.extra_answer_frac=0.5", "rollout.exclude_benchmark_answers=True"])
    train = [w for w in get_split("train") if w not in bench]
    extra = [w for w in extra_answer_words() if w not in bench]
    for step in range(50):
        assert not {a for a, _ in plan_step(cfg, step, train, extra)} & bench


def test_sft_tokenize_skips_benchmark_answers(tiny_model_dir, tiny_traces):
    from fiveletters.train.sft import _tokenize

    tok = load_model_and_tokenizer(tiny_model_dir, "cpu", for_training=False)[1]
    answers = {json.loads(line)["answer"] for line in open(tiny_traces / "shard-00000.jsonl")}
    full, _, _ = _tokenize(tok, [tiny_traces], 512, 0)
    some = frozenset(sorted(answers)[:3])
    kept, _, _ = _tokenize(tok, [tiny_traces], 512, 0, some)
    assert 0 < len(kept) < len(full)


def test_hub_model_resolves_from_local_copy_first(tiny_model_dir, tmp_path, monkeypatch):
    import shutil as sh

    from fiveletters.train.common import export_model, resolve_model_path

    heavy = tmp_path / "heavy"
    sh.copytree(tiny_model_dir, heavy / "sft-x" / "best")
    monkeypatch.setenv("FL_HEAVY_DIR", str(heavy))
    # resolves without touching the Hub (the repo id does not even exist)
    assert resolve_model_path("hub:nobody/does-not-exist:sft-x/best") == str(heavy / "sft-x" / "best")

    monkeypatch.setenv("FL_MODEL_EXPORT_DIR", str(tmp_path / "export"))
    export_model(heavy / "sft-x" / "best", "sft-x", "best")
    assert (tmp_path / "export" / "sft-x" / "best" / "config.json").exists()
    monkeypatch.delenv("FL_HEAVY_DIR")
    assert resolve_model_path("hub:nobody/does-not-exist:sft-x/best") == str(tmp_path / "export" / "sft-x" / "best")


def test_missing_model_fails_clearly(tmp_path, monkeypatch):
    from fiveletters.train import common

    monkeypatch.delenv("FL_HEAVY_DIR", raising=False)
    monkeypatch.delenv("FL_MODEL_EXPORT_DIR", raising=False)
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *a, **k: str(tmp_path))
    with pytest.raises(FileNotFoundError, match="neither local nor on the Hub"):
        common.resolve_model_path("hub:nobody/repo:run/best")


def test_consistency_penalty_lowers_reward():
    from fiveletters.game import Game
    from fiveletters.words import load

    vw = load().guess_set
    good, bad = Game("agile"), Game("agile")
    for w in ("crate", "mails", "anile", "agile"):
        good.step("", w, vw)
    for w in ("crate", "mails", "anile", "apian", "agile"):
        bad.step("", w, vw)
    cfg = {"win": 1.0, "speed_bonus": 0.0, "invalid_penalty": 0.2, "consistency_penalty": 0.3}
    assert game_reward(bad, cfg) < game_reward(good, cfg)


def test_v5_plan_commands_parse_and_configs_load():
    """Every command in the v5 plan must parse and every --set key must exist, so a typo cannot burn a session."""
    import shlex
    from pathlib import Path

    from fiveletters.cli import build_parser
    from fiveletters.train import grpo, sft

    root = Path(__file__).resolve().parents[1]
    plans = {f"{f.stem}:{k}": v for f in (root / "kaggle" / "plans").glob("*.json")
             for k, v in json.loads(f.read_text()).items()}
    n = 0
    for name, sess in plans.items():
        if name.split(":")[1].startswith("_"):
            continue
        for cmd in sess["gpu0"] + sess["gpu1"]:
            args = build_parser().parse_args(shlex.split(cmd.replace("FINAL_RUN", "x").replace("BASE_RUN", "x")))
            if args.cmd in ("sft", "grpo"):
                defaults = sft.DEFAULTS if args.cmd == "sft" else grpo.DEFAULTS
                load_config(defaults, str(root / args.config), args.set)
            n += 1
    assert n > 20


def test_grpo_with_v5_settings_runs(tiny_model_dir, tmp_path):
    from fiveletters.train.grpo import DEFAULTS, train_grpo

    cfg = load_config(DEFAULTS, None, [
        f"model.init={tiny_model_dir}", f"run.runs_dir={tmp_path}", "run.name=t-grpo-v5", "curriculum.teacher=entropy",
        "curriculum.consistent=True", "reward.consistency_penalty=0.2", "rollout.extra_answer_frac=0.5",
        "rollout.exclude_benchmark_answers=True", "eval.split=select",
        "rollout.answers_per_step=2", "rollout.group_size=2", "rollout.max_new_tokens=12",
        "train.steps=2", "train.micro_batch_size=4", "eval.every_steps=2", "eval.games=2", "eval.batch_size=2",
        "checkpoint.every_minutes=0", "budget.hours=none", "budget.reserve_minutes=0",
    ])
    assert train_grpo(cfg) == 0
    logged = [json.loads(x) for x in (tmp_path / "t-grpo-v5" / "metrics.jsonl").read_text().splitlines() if '"reward"' in x]
    assert len(logged) == 2 and all("clue_contradiction_rate" in r for r in logged)


def test_distill_plays_only_training_answers_and_keeps_won_model_turns(tmp_path):
    from fiveletters.distill import distill_answers, generate, samples_from_game
    from fiveletters.game import Game
    from fiveletters.policies.base import SolverPolicy
    from fiveletters.solvers import make_solver
    from fiveletters.words import benchmark_words, get_split

    words = set(distill_answers())
    never = set(get_split("select")) | set(get_split("heldout")) | set(get_split("val")) | benchmark_words()
    assert words and not words & never

    policy = SolverPolicy(make_solver("entropy", pool="all"))  # extra answers lie outside the answer list
    out = tmp_path / "d"
    # two parts on two "GPUs": the manifest appears only when every shard exists
    r0 = generate(policy, out, openers=(None, "slate"), shard_size=3, part=0, parts=2, limit=6)
    assert not r0["complete"] and not (out / "manifest.json").exists()
    r1 = generate(policy, out, openers=(None, "slate"), shard_size=3, part=1, parts=2, limit=6)
    assert r1["complete"] and json.loads((out / "manifest.json").read_text())["shards"] == 2
    tcfg = json.loads((out / "trace_config.json").read_text())  # sft reads the prompt format from here
    assert tcfg["spaced"] is True and tcfg["state_line"] is True
    rows = [json.loads(x) for p in sorted(out.glob("shard-*.jsonl")) for x in p.read_text().splitlines()]
    assert rows and all(r["game_won"] and r["answer"] in words for r in rows)
    # forced opener turns are context only: no sample has SLATE as its target on turn 1 of an opener game
    assert all(not (r["turn"] == 1 and "S L A T E" in r["completion"]) for r in rows)
    assert generate(policy, out, openers=(None, "slate"), shard_size=3, part=0, parts=2, limit=6)["skipped"] == 1
    lost = Game("ghost")
    assert samples_from_game(lost, 0) == []


def test_replay_subsample_is_deterministic_and_close_to_the_fraction():
    from fiveletters.train.sft import _keep

    rows = [{"answer": f"w{i:04d}", "game": i, "turn": 1, "completion": "<guess>x</guess>"} for i in range(5000)]
    kept = [r for r in rows if _keep(r, 0.1)]
    assert kept == [r for r in rows if _keep(r, 0.1)]  # same subset every time
    assert 400 < len(kept) < 600 and all(_keep(r, 1.0) for r in rows[:10])
