"""Round v6: common-word + probe teacher, whole-word guess format, spelling drill, trace conversion, token-trie helper."""

import json

import pytest

from fiveletters.feedback import feedback
from fiveletters.game import Game, Turn, fits_clues
from fiveletters.prompts import PromptFormat, parse_guess, render_completion, render_prompt
from fiveletters.words import benchmark_words, get_split, load

WL = load()


def _play(solver, answer):
    g = Game(answer)
    while not g.done:
        w = solver.choose(g.turns)
        g.step(f"<guess>{w}</guess>", w, WL.guess_set)
    return [t.guess for t in g.turns], g.won


def test_teacher_probes_on_traps_and_prefers_common_words():
    from fiveletters.solvers import make_solver

    guesses, won = _play(make_solver("common-probe", vocab="shared"), "watch")
    assert won and len(guesses) <= 4
    assert any(g not in ("batch", "hatch", "latch", "match", "patch", "watch") for g in guesses[1:-1])  # a probe
    # with two candidates left, the more common English word comes first
    s = make_solver("common-only", pool="all", vocab="wordle")
    turns = [Turn("", "aloft", True, tuple(feedback("aloft", "atone"))),
             Turn("", "crane", True, tuple(feedback("crane", "atone"))),
             Turn("", "dumbs", True, tuple(feedback("dumbs", "atone")))]
    cands = [s.pool[c] for c in s.candidates(turns)]
    if len(cands) <= 2 and "atone" in cands:
        assert s.choose(turns) == "atone"
    # every non-probe guess fits all clues
    g = Game("craze")
    t = make_solver("common-only", vocab="shared")
    while not g.done:
        w = t.choose(g.turns)
        assert fits_clues(w, g.turns)
        g.step(f"<guess>{w}</guess>", w, WL.guess_set)


def test_whole_word_format_keeps_spaced_history():
    fmt = PromptFormat(guess_word=True)
    g = Game("ghost")
    g.step("", "crate", WL.guess_set)
    comp = render_completion(g, "ghost", fmt)
    assert comp.endswith("<guess>ghost</guess>") and parse_guess(comp) == "ghost"
    prompt = render_prompt(g, fmt)
    assert "Turn 1: C R A T E ->" in prompt and "<guess>crane</guess>" in prompt


def test_convert_only_touches_guess_tags():
    from fiveletters.drill import to_guess_word

    s = "Reply with <state>x</state> then <guess>C R A N E</guess>.\nTurn 1: C R A T E -> X X G X X"
    assert to_guess_word(s) == "Reply with <state>x</state> then <guess>crane</guess>.\nTurn 1: C R A T E -> X X G X X"
    assert to_guess_word("<state>fixed: _ _ A _ _</state>\n<guess>S L A N T</guess>").endswith("<guess>slant</guess>")


def test_drill_is_fair_and_targets_fit(tmp_path):
    from fiveletters.drill import generate_drill, write_manifest

    fmt = PromptFormat(guess_word=True)
    res = generate_drill(tmp_path, 40, fmt, seed=3, shard_size=25)
    (tmp_path / "trace_config.json").write_text(json.dumps({"spaced": True, "state_line": True, "guess_word": True}))
    assert write_manifest(tmp_path)["samples"] == res["samples"] == 40
    never = (set(get_split("val")) | set(get_split("heldout")) | set(get_split("select")) | set(get_split("rare"))
             | benchmark_words())
    rows = [json.loads(x) for p in sorted(tmp_path.glob("shard-drill-*.jsonl")) for x in p.read_text().splitlines()]
    for r in rows:
        assert r["answer"] not in never  # no drill board is built around a dev or test word
        target = parse_guess(r["completion"])
        assert target in WL.guess_set
        # rebuild the board from the prompt and check the target fits every clue
        turns = []
        for line in r["prompt"].splitlines():
            if line.startswith("Turn "):
                word = line.split(": ", 1)[1].split(" ->")[0].replace(" ", "").lower()
                turns.append(Turn("", word, True, tuple(feedback(word, r["answer"]))))
        assert turns and fits_clues(target, turns) and target not in {t.guess for t in turns}


def test_saved_format_is_read_back(tmp_path, tiny_model_dir):
    pytest.importorskip("transformers")
    from fiveletters.lm import load_model_and_tokenizer
    from fiveletters.prompts import load_format
    from fiveletters.train.common import atomic_save_model

    model, tok = load_model_and_tokenizer(tiny_model_dir, "cpu", for_training=False)
    fmt = PromptFormat(guess_word=True)
    assert atomic_save_model(model, tok, tmp_path / "best", fmt)
    assert load_format(tmp_path / "best") == fmt
    assert load_format(tiny_model_dir) is None


class _WordTok:
    """Tiny tokenizer: words split into known chunks (longest first); '</' closes the tag."""

    chunks = ["</", "guess>", "at", "one", "ax", "ag", "a", "t", "o", "n", "e", "x", "g", "z", "h"]

    name_or_path = "wordtok"

    def __len__(self):
        return len(self.chunks) + 1

    def __call__(self, text, add_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            for j, c in enumerate(self.chunks):
                if text.startswith(c, i):
                    ids.append(j)
                    i += len(c)
                    break
            else:
                raise ValueError(text)
        return {"input_ids": ids}


def test_token_trie_search_matches_brute_force():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from fiveletters.constrain import best_dictionary_word_tokens

    tok = _WordTok()
    torch.manual_seed(1)
    cfg = transformers.LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    model = transformers.LlamaForCausalLM(cfg).eval()
    words = frozenset({"atone", "axone", "agone", "ozone", "thaze"})
    ctx = [len(tok) - 1, 0]
    close = tok("</guess>"[:2])["input_ids"][0]

    def exact(w):
        ids = tok(w)["input_ids"] + [close]
        lp = torch.log_softmax(model(input_ids=torch.tensor([ctx + ids])).logits[0].float(), -1)
        return sum(float(lp[len(ctx) - 1 + k, t]) for k, t in enumerate(ids))

    scores = {w: exact(w) for w in words}
    best = max(scores, key=scores.get)
    word, score = best_dictionary_word_tokens(model, tok, ctx, words, beams=64)
    assert word == best and abs(score - scores[best]) < 1e-4
    second = sorted(scores, key=scores.get)[-2]
    assert best_dictionary_word_tokens(model, tok, ctx, words, exclude={best}, beams=64)[0] == second
