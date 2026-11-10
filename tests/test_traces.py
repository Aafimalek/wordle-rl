import json

import pytest

from fiveletters import traces as T
from fiveletters import words
from fiveletters.prompts import parse_guess
from fiveletters.solvers import EntropySolver


@pytest.fixture(autouse=True)
def fast_teachers(monkeypatch):
    """The real teacher is the lookahead solver; entropy keeps these tests quick."""
    T._WORKER.clear()
    main, fast = EntropySolver(), EntropySolver()
    main._memo_loaded = 1  # skip warm-up
    main.save_memo = fast.save_memo = lambda: None
    monkeypatch.setattr(T, "_teachers", lambda cfg: (main, fast, main.ranked_openers(5)))
    yield
    T._WORKER.clear()


def test_teacher_game_targets_are_teacher_moves():
    cfg = T.TraceConfig(seed=7)
    for gi in range(10):
        samples = T.play_teacher_game("ghost", gi, cfg)
        assert samples, "every game has at least one target turn"
        for s in samples:
            assert parse_guess(s["completion"]) is not None
            assert s["prompt"].count("Turn ") == s["turn"] - 1


def test_traces_are_deterministic_and_resumable(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "get_split", lambda name: list(words._split_cached("train")[:6]))
    cfg = T.TraceConfig(games_per_word=2, words_per_shard=2, seed=3)
    m1 = T.generate_traces(tmp_path / "a", cfg)
    assert m1["shards"] == 3
    shard = sorted((tmp_path / "a").glob("shard-*.jsonl"))[1]
    original = shard.read_text()
    shard.unlink()
    (tmp_path / "a" / "shard-00001.jsonl.tmp").write_text("half written")
    m2 = T.generate_traces(tmp_path / "a", cfg)
    assert shard.read_text() == original and m2["samples"] == m1["samples"]
    assert not list((tmp_path / "a").glob("*.tmp"))
    with pytest.raises(RuntimeError, match="different config"):
        T.generate_traces(tmp_path / "a", T.TraceConfig(games_per_word=3, words_per_shard=2, seed=3))


def test_no_eval_answers_in_train_traces():
    heldout = set(words.get_split("heldout")) | set(words.get_split("val"))
    assert not heldout & set(words.get_split("train"))
    s = T.play_teacher_game(words.get_split("train")[0], 0, T.TraceConfig())
    assert json.dumps(s)  # serializable
