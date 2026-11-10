"""Round v7: balanced spelling drill, replay mixing, and the retry-on-rejected-word rule."""

import json
from collections import Counter

import pytest

from fiveletters.feedback import feedback
from fiveletters.game import Game, Turn, fits_clues
from fiveletters.prompts import PromptFormat, parse_guess, render_prompt
from fiveletters.words import benchmark_words, load

WL = load()
FMT = PromptFormat()


def _rows(folder, pattern="shard-*.jsonl"):
    return [json.loads(x) for p in sorted(folder.glob(pattern)) for x in p.read_text().splitlines()]


def _board(row):
    """Valid turns of the board in a prompt, with feedback recomputed against the row's hidden word."""
    turns = []
    for line in row["prompt"].splitlines():
        if line.startswith("Turn ") and "INVALID" not in line:
            word = line.split(": ", 1)[1].split(" ->")[0].replace(" ", "").lower()
            turns.append(Turn("", word, True, tuple(feedback(word, row["answer"]))))
    return turns


@pytest.mark.parametrize("strict", [False, True])
def test_balanced_drill_is_fair_and_balanced(tmp_path, strict):
    from fiveletters.drill import DRILL_ZIPF_MIN, dev_test_words, generate_balanced_drill
    from fiveletters.solvers import word_zipf

    res = generate_balanced_drill(tmp_path, 4, FMT, seed=5, shard_size=100, strict=strict, limit_words=60,
                                  invalid_prob=0.3)
    rows = _rows(tmp_path)
    never = dev_test_words()
    assert res["samples"] == len(rows) == 60 * 4 - res["skipped"] and res["skipped"] <= 4
    counts = Counter()
    for r in rows:
        assert r["answer"] not in never  # no board is built around a dev or test word
        target = parse_guess(r["completion"])
        counts[target] += 1
        turns = _board(r)
        assert turns and target in WL.guess_set and fits_clues(target, turns)
        assert target not in {t.guess for t in turns}
        assert r["completion"].endswith("<guess>" + " ".join(target.upper()) + "</guess>")  # spaced letters
        if strict:
            assert target not in never
    assert max(counts.values()) <= 4 and min(counts.values()) >= 3  # every word the same number of times
    assert float(word_zipf(list(counts)).min()) >= DRILL_ZIPF_MIN
    assert any("INVALID (not in word list)" in r["prompt"] for r in rows)
    if not strict:
        assert any(t in never for t in counts)  # the disclosed choice: dev/test words can be spelling targets
    # same seed, same data
    again = tmp_path / "again"
    generate_balanced_drill(again, 4, FMT, seed=5, shard_size=100, strict=strict, limit_words=60, invalid_prob=0.3)
    assert _rows(again) == rows


def _fake_traces(folder, answers, per_answer=40):
    folder.mkdir()
    (folder / "trace_config.json").write_text(json.dumps({"spaced": True, "state_line": True, "teacher": "t"}))
    with open(folder / "shard-00000.jsonl", "w", encoding="utf-8") as f:
        for a in answers:
            for k in range(per_answer):
                f.write(json.dumps({"prompt": "p", "completion": f"<guess>{a}</guess>", "answer": a, "game": k,
                                    "turn": 1, "game_won": True}) + "\n")


def test_replay_is_deterministic_and_benchmark_clean(tmp_path):
    from fiveletters.drill import build_v7, copy_replay

    bench = sorted(benchmark_words())[0]
    train = [w for w in WL.answers if w not in benchmark_words()][:5]
    _fake_traces(tmp_path / "src", [*train, bench])
    a = copy_replay(tmp_path / "src", tmp_path / "a", 0.5, "shard-replay0")
    b = copy_replay(tmp_path / "src", tmp_path / "b", 0.5, "shard-replay0")
    rows = _rows(tmp_path / "a")
    assert rows == _rows(tmp_path / "b") and a["samples"] == b["samples"] == len(rows)
    assert 60 <= len(rows) <= 140 and all(r["answer"] != bench for r in rows)  # about half of 200, no leaderboard word
    assert copy_replay(tmp_path / "src", tmp_path / "c", 1.0, "shard-replay0")["samples"] == 200

    res = build_v7(tmp_path / "v7", 2, [(str(tmp_path / "src"), 0.5)], seed=2, limit_words=20)
    cfg = json.loads((tmp_path / "v7" / "trace_config.json").read_text())
    assert (cfg["spaced"], cfg["state_line"], cfg["guess_word"]) == (True, True, False)
    manifest = json.loads((tmp_path / "v7" / "manifest.json").read_text())
    assert manifest["samples"] == res["samples"] == res["drill"]["samples"] + len(rows)
    with pytest.raises(FileExistsError):
        build_v7(tmp_path / "v7", 2, [], seed=2, limit_words=20)


def test_retry_rule_rejects_without_using_a_turn():
    g = Game("atone", retries=2)
    g.step("<guess>ABONE</guess>", "abone", WL.guess_set)  # rejected, free
    assert len(g.turns) == 1 and g.turns_left == 6 and not g.turns[0].valid and g.turns[0].free
    g.step("<guess>AXXNE</guess>", "axxne", WL.guess_set)  # rejected, free (second and last retry of this turn)
    assert g.turns_left == 6 and g.turns[1].free
    g.step("<guess>AZZNE</guess>", "azzne", WL.guess_set)  # retries used up: this one costs the turn
    assert g.turns_left == 5 and not g.turns[2].free
    g.step("<guess>CRATE</guess>", "crate", WL.guess_set)
    assert g.turns_left == 4
    g.step("<guess>QQQQQ</guess>", "qqqqq", WL.guess_set)  # a new turn has its own retries
    assert g.turns_left == 4 and g.turns[4].free
    prompt = render_prompt(g, FMT)
    assert "Turn 1: A B O N E -> INVALID (not in word list)" in prompt and "Turns left: 4" in prompt
    g.step("<guess>ATONE</guess>", "atone", WL.guess_set)
    rec = g.record()
    assert g.won and rec["n_turns"] == 3 and rec["n_invalid"] == 1 and rec["n_rejected"] == 3
    assert rec["retries"] == 2

    strict = Game("atone")
    strict.step("<guess>ABONE</guess>", "abone", WL.guess_set)
    assert strict.turns_left == 5 and not strict.turns[0].free and "n_rejected" not in strict.record()


def test_retry_rule_always_ends():
    g = Game("atone", retries=3)
    n = 0
    while not g.done:
        g.step("<guess>ABONE</guess>", "abone", WL.guess_set)
        n += 1
    assert n == 6 * 4 and not g.won and g.record()["n_turns"] == 6


def test_leaderboard_rule_scores_any_five_letters():
    g = Game("atone", open_vocab=True)
    t = g.step("<guess>ABONE</guess>", "abone", WL.guess_set)  # not a word: scored anyway, uses the turn
    assert t.valid and t.nonword and tuple(t.feedback) == tuple(feedback("abone", "atone")) and g.turns_left == 5
    assert "Turn 1: A B O N E -> " in render_prompt(g, FMT) and "INVALID" not in render_prompt(g, FMT)
    for bad in ("abon", "abones", "ab0ne"):  # not five letters a-z: still an invalid turn
        assert not g.step("x", bad, WL.guess_set).valid
    assert not g.step("no tag", None, WL.guess_set).valid
    g.step("<guess>ATONE</guess>", "atone", WL.guess_set)
    rec = g.record()
    assert g.won and rec["open_vocab"] and rec["n_nonword"] == 1 and rec["n_invalid"] == 4 and rec["n_turns"] == 6
    assert "nonword" not in rec["turns"][-1] and rec["turns"][0]["nonword"]

    strict = Game("atone")
    assert not strict.step("<guess>ABONE</guess>", "abone", WL.guess_set).valid
    assert "open_vocab" not in strict.record() and "nonword" not in strict.record()["turns"][0]


def test_leaderboard_rule_eval_and_audit(tmp_path):
    import importlib.util
    from pathlib import Path

    from fiveletters.eval.harness import run_eval
    from fiveletters.policies.base import Policy

    class Typo(Policy):
        name = "typo"

        def act(self, games):  # first a near-miss non-word, then the answer
            return [f"<guess>{('q' + g.answer[1:]) if not g.turns else g.answer}</guess>" for g in games]

    words = [w for w in WL.answers if "q" + w[1:] not in WL.guess_set][:20]
    s = run_eval(Typo(), words, tmp_path / "open", word_set="custom", open_vocab=True, batch_size=8)
    assert s["rule"] == "open-vocab" and s["win_rate"] == 1.0 and s["invalid_turn_rate"] == 0.0
    assert s["nonword_guess_rate"] == 0.5 and s["games_with_nonword"] == 20
    strict = run_eval(Typo(), words, tmp_path / "strict", word_set="custom", batch_size=8)
    assert "rule" not in strict and strict["invalid_turn_rate"] == 0.5 and "nonword_guess_rate" not in strict
    with pytest.raises(ValueError):
        run_eval(Typo(), words, tmp_path / "both", word_set="custom", open_vocab=True, retries=2)

    spec = importlib.util.spec_from_file_location(
        "audit_eval_v7", Path(__file__).resolve().parents[1] / "scripts" / "audit_eval.py")
    audit_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit_eval)
    assert audit_eval.audit(tmp_path / "open", WL.guess_set) == []
    assert audit_eval.audit(tmp_path / "strict", WL.guess_set) == []
    # the same games presented as strict results must be caught: a non-word was accepted
    summ = json.loads((tmp_path / "open" / "summary.json").read_text())
    summ.pop("open_vocab")
    (tmp_path / "open" / "summary.json").write_text(json.dumps(summ))
    assert any("not in the official guess list" in p for p in audit_eval.audit(tmp_path / "open", WL.guess_set))


def test_deep_drill_boards(tmp_path):
    from fiveletters.drill import DEEP_DEPTHS, dev_test_words, generate_balanced_drill

    res = generate_balanced_drill(tmp_path, 3, FMT, seed=8, shard_size=1000, limit_words=80, depths=DEEP_DEPTHS,
                                  invalid_prob=0.5, max_invalid=2)
    rows = _rows(tmp_path)
    never = dev_test_words()
    assert res["samples"] == len(rows) and res["max_invalid"] == 2 and "5" in res["depths"]
    n_bad = []
    for r in rows:
        assert r["answer"] not in never
        target = parse_guess(r["completion"])
        turns = _board(r)
        assert fits_clues(target, turns) and 1 <= len(turns) <= 5
        bad = r["prompt"].count("INVALID (not in word list)")
        n_bad.append(bad)
        assert len(turns) + bad <= 5 and r["turn"] == len(turns) + bad + 1 <= 6  # the target always has a turn
    assert max(n_bad) == 2 and sum(len(_board(r)) >= 4 for r in rows) > len(rows) / 10  # v7 boards: 6%
    with pytest.raises(ValueError):
        generate_balanced_drill(tmp_path / "x", 1, FMT, limit_words=2, depths=((6, 1.0),))


def test_recovery_samples_from_the_models_own_rejected_words(tmp_path):
    from fiveletters.drill import dev_test_words
    from fiveletters.policies.base import Policy
    from fiveletters.recovery import generate, recovery_hidden_words

    never = dev_test_words()
    pool = recovery_hidden_words(40)
    assert len(pool) == 40 and not set(pool) & never and not set(pool) & set(WL.answers)

    class Stuck(Policy):
        """Opens with CRATE, then repeats one non-word until the game ends: the failure seen in v8."""
        name = "stuck"
        fmt = FMT

        def describe(self):
            return {"policy": "stuck"}

        def act(self, games):
            return ["<guess>C R A T E</guess>" if not g.turns else "<guess>Q Z X J V</guess>" for g in games]

    res = generate(Stuck(), tmp_path, limit=40, part=1, parts=2, per_event=3, repeat_prob=0.5, seed=3)
    rows = _rows(tmp_path)
    games = [json.loads(x) for x in (tmp_path / "games.jsonl").read_text().splitlines()]
    assert res["games"] == len(games) == 20 and [g["answer"] for g in games] == pool[1::2]
    # one event per game (the repeats of the same word are not new events), at most 3 targets each
    assert res["events"] == 20 and res["nonwords_first_time"] == 20 and res["nonwords_repeated"] == 80
    assert res["samples"] == len(rows) and 20 <= len(rows) <= 60
    twice = 0
    for r in rows:
        assert r["answer"] not in never and r["rejected"] == "qzxjv" and r["source"] == "recovery"
        n_bad = r["prompt"].count("Q Z X J V -> INVALID (not in word list)")
        assert n_bad in (1, 2) and r["turn"] == 2 + n_bad
        twice += n_bad == 2
        target = parse_guess(r["completion"])
        assert target in WL.guess_set and target != "crate" and fits_clues(target, _board(r))
    assert 0 < twice < len(rows)
    by_game = Counter(r["game"] for r in rows)
    assert max(by_game.values()) <= 3
    for g, n in by_game.items():  # the targets of one event are different words
        assert len({r["completion"] for r in rows if r["game"] == g}) == n
    cfg = json.loads((tmp_path / "trace_config.json").read_text())
    assert cfg["source"] == "v10-recovery" and cfg["spaced"] and cfg["state_line"] and cfg["events"] == 20
    assert json.loads((tmp_path / "manifest.json").read_text())["samples"] == len(rows)
    with pytest.raises(FileExistsError):
        generate(Stuck(), tmp_path, limit=4)

    from fiveletters.recovery import recovery_samples
    with pytest.raises(ValueError):  # a dev or test answer can never be the hidden word
        import random
        recovery_samples(Game(sorted(never)[0]), 0, [], FMT, random.Random(0))
