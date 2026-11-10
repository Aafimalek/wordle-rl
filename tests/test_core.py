import random

import numpy as np
import pytest

from fiveletters import feedback as F
from fiveletters import words
from fiveletters.game import ERR_FORMAT, ERR_HARD_MODE, ERR_NOT_A_WORD, Game, hard_mode_violation
from fiveletters.patterns import pattern_matrix
from fiveletters.prompts import PromptFormat, parse_guess, parse_state, render_completion, render_prompt
from fiveletters.state import state_line


@pytest.mark.parametrize("guess,answer,expected", [
    ("salet", "ghost", "YXXXG"),
    ("moist", "ghost", "XYXGG"),
    ("ghost", "ghost", "GGGGG"),
    ("speed", "abide", "XXYXY"),  # first E yellow, second E grey: the answer has one E
    ("eerie", "theme", "YXXXG"),  # last E green first, then one more E available for the first E
    ("level", "hello", "YGXXY"),  # E green; HELLO has two Ls, so both Ls are yellow
    ("llama", "hello", "YYXXX"),
    ("allee", "eagle", "YYXYG"),
    ("aaaaa", "abbey", "GXXXX"),
])
def test_feedback_repeated_letters(guess, answer, expected):
    assert F.to_str(F.feedback(guess, answer)) == expected


def test_encode_roundtrip():
    for code in range(243):
        assert F.encode(F.decode(code)) == code
    assert F.encode(F.ALL_GREEN) == F.ALL_GREEN_CODE == 242


def test_vectorized_matrix_matches_rule():
    wl = words.load()
    rng = random.Random(1)
    g = rng.sample(wl.guesses, 300) + ["speed", "eerie", "llama", "aaaaa"]
    a = rng.sample(wl.answers, 300) + ["abide", "theme", "hello", "abbey"]
    m = pattern_matrix(g, a)
    want = np.array([[F.encode(F.feedback(x, y)) for y in a] for x in g], dtype=np.uint8)
    assert np.array_equal(m, want)


def test_word_lists_and_splits_are_frozen():
    wl = words.load()
    assert (len(wl.answers), len(wl.extra), len(wl.guesses)) == (2315, 10657, 12972)
    words.check_resources()
    sizes = {n: len(words.get_split(n)) for n in words.SPLIT_NAMES}
    assert sizes == {"train": 1752, "val": 100, "heldout": 463, "full": 2315, "rare": 500}
    assert not set(words.get_split("rare")) & set(wl.answers)
    # regenerating from the seed gives exactly the committed files
    assert words.make_splits(wl) == {n: words.get_split(n) for n in words.SPLIT_NAMES}


def test_state_line_example_from_plan():
    g = Game("ghost")
    for w in ("salet", "moist"):
        g.step("", w, words.load().guess_set)
    assert state_line(g.turns) == "fixed: _ _ _ S T | has: O(not 2) | out: A E I L M"


def test_state_line_tracks_duplicate_letters():
    g = Game("geese")
    g.step("", "eerie", words.load().guess_set)  # answer has three Es
    line = state_line(g.turns)
    assert line.startswith("fixed: _ E _ _ E") and "E(not 1)" in line


def test_strict_turns_and_game_end():
    vw = words.load().guess_set
    g = Game("ghost")
    assert g.step("junk", None, vw).error == ERR_FORMAT
    assert g.step("<guess>QZZZZ</guess>", "qzzzz", vw).error == ERR_NOT_A_WORD
    for _ in range(4):
        g.step("", "salet", vw)
    assert g.done and not g.won and g.n_invalid == 2
    with pytest.raises(RuntimeError):
        g.step("", "ghost", vw)


def test_hard_mode():
    vw = words.load().guess_set
    g = Game("ghost", hard_mode=True)
    g.step("", "salet", vw)  # S yellow, T green
    assert hard_mode_violation("crane", g.turns)
    assert hard_mode_violation("toast", g.turns) is None
    assert g.step("", "crane", vw).error == ERR_HARD_MODE


@pytest.mark.parametrize("text,want", [
    ("<guess>G H O S T</guess>", "ghost"),
    ("<state>x</state>\n<guess>ghost</guess>", "ghost"),
    ("<guess>CRANE</guess> then <guess>G H O S T</guess>", "ghost"),
    ("<guess>G H O S</guess>", None),
    ("<guess>GH0ST</guess>", None),
    ("GHOST", None),
    ("", None),
])
def test_parse_guess(text, want):
    assert parse_guess(text) == want


def test_prompt_and_completion_roundtrip():
    vw = words.load().guess_set
    g = Game("ghost")
    g.step("", "salet", vw)
    g.step("x", None, vw)
    p = render_prompt(g)
    assert "Turn 1: S A L E T -> Y X X X G" in p and "Turn 2: (none) -> INVALID" in p and "Turns left: 4" in p
    c = render_completion(g, "moist")
    assert parse_guess(c) == "moist" and parse_state(c) == state_line(g.turns)
    plain = PromptFormat(spaced=False, state_line=False)
    assert "SALET -> YXXXG" in render_prompt(g, plain)
    assert render_completion(g, "moist", plain) == "<guess>MOIST</guess>"


def test_clue_contradictions_counted():
    from fiveletters.game import count_clue_contradictions, fits_clues

    vw = words.load().guess_set
    g = Game("agile")
    for w in ("crate", "mails", "anile", "apian"):  # APIAN contradicts ANILE -> G X G G G
        g.step("", w, vw)
    assert fits_clues("agile", g.turns) and not fits_clues("apian", g.turns[:3])
    bad, checked = count_clue_contradictions(g.turns)
    assert checked == 3 and bad >= 1


def test_metrics_report_contradictions():
    from fiveletters.eval.metrics import summarize

    vw = words.load().guess_set
    g = Game("ghost")
    for w in ("salet", "moist", "ghost"):
        g.step("", w, vw)
    s = summarize([g.record()])
    assert "clue_contradiction_rate" in s


def test_gate_check_and_wait(tmp_path, monkeypatch):
    import json

    from fiveletters.cli import main

    monkeypatch.setenv("FL_RUNS_DIR", str(tmp_path))
    d = tmp_path / "eval" / "heldout" / "m"
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps({"games": 463, "win_rate": 0.81}))

    def rc(argv):
        try:
            main(argv)
        except SystemExit as e:
            return e.code
    assert rc(["gate", "g1", "--check", "heldout/m", "--min-win", "0.795"]) == 0
    assert rc(["gate", "g1", "--wait-minutes", "0"]) == 0          # the other queue sees the decision
    assert rc(["gate", "g2", "--check", "heldout/m", "--min-win", "0.81"]) == 76  # strictly above
    assert rc(["gate", "g2", "--wait-minutes", "0"]) == 76
    assert rc(["gate", "g3", "--check", "heldout/missing", "--min-win", "0.1"]) == 76
    assert rc(["gate", "never", "--wait-minutes", "0"]) == 76      # no decision -> stop, never run blind


def test_select_split_is_never_trained_and_benchmark_free():
    from fiveletters.words import benchmark_words, extra_answer_words, get_split

    sel = set(get_split("select"))
    assert len(sel) == 534
    assert not sel & set(get_split("train")) and not sel & benchmark_words()
    assert not sel & set(extra_answer_words(broad=True))
