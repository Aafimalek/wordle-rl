"""Dictionary-constrained decoding: only real words can be typed inside <guess>."""

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from fiveletters.constrain import GuessConstraint  # noqa: E402


class _Tok:
    """A tiny stand-in tokenizer: single letters, ' X' letters, a two-letter token and the closing tag."""

    def __init__(self):
        self.vocab = ["<guess>", "</", "guess>", "\n", "x"] + [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"] + \
                     [" " + c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"] + ["HO"]
        self.name_or_path = "fake"

    def __len__(self):
        return len(self.vocab)

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.vocab[i] for i in ids)

    def id(self, s):
        return self.vocab.index(s)


def test_only_word_prefixes_allowed():
    tok = _Tok()
    c = GuessConstraint(tok, frozenset({"ghost", "ghoul"}), prompt_len=0)
    assert c.allowed("no tag yet") is None
    allowed = {tok.vocab[i].strip() for i in c.allowed("<guess>")}
    assert allowed == {"G"}
    allowed = {tok.vocab[i].strip() for i in c.allowed("<guess>G")}
    assert allowed == {"H", "HO"}  # multi-letter tokens are fine when G+HO = GHO is still a word prefix
    allowed = {tok.vocab[i].strip() for i in c.allowed("<guess>G H O")}
    assert allowed == {"S", "U"}
    assert {tok.vocab[i] for i in c.allowed("<guess>G H O S T")} == {"</"}  # five letters: close the tag
    assert c.allowed("<guess>G H O S T</") is None  # closing: no constraint


def test_masks_scores():
    tok = _Tok()
    c = GuessConstraint(tok, frozenset({"ghost"}), prompt_len=1)
    ids = torch.tensor([[tok.id("x"), tok.id("<guess>"), tok.id("G")]])
    scores = torch.zeros(1, len(tok))
    out = c(ids, scores)
    finite = {tok.vocab[i] for i in torch.isfinite(out[0]).nonzero().flatten().tolist()}
    assert finite == {"H", " H", "HO"}


class _CallTok(_Tok):
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [self.id(text)]}


def _tiny_lm(vocab_size):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=vocab_size, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    return transformers.LlamaForCausalLM(cfg).eval()


@torch.no_grad()
def _exact_logprob(model, tok, ctx, word):
    ids = [tok.id(word[0])] + [tok.id(" " + c) for c in word[1:]]
    logits = model(input_ids=torch.tensor([ctx + ids])).logits[0]
    lp = torch.log_softmax(logits.float(), -1)
    return sum(float(lp[len(ctx) - 1 + k, t]) for k, t in enumerate(ids))


def test_word_search_finds_the_models_best_whole_word():
    from fiveletters.constrain import best_dictionary_word

    tok = _CallTok()
    model = _tiny_lm(len(tok))
    words = frozenset({"ghost", "ghoul", "crate", "slate", "plant", "glyph", "chant", "shout", "about", "zebra"})
    ctx = [tok.id("x"), tok.id("<guess>")]
    exact = {w: _exact_logprob(model, tok, ctx, w.upper()) for w in words}
    best = max(exact, key=exact.get)
    word, score = best_dictionary_word(model, tok, ctx, words, beams=64)  # wide beam = exhaustive here
    assert word == best.upper() and abs(score - exact[best]) < 1e-4
    # an excluded word is never returned, and the next best one is
    second = sorted(exact, key=exact.get)[-2]
    assert best_dictionary_word(model, tok, ctx, words, exclude={best}, beams=64)[0] == second.upper()
    # a narrow beam still returns a real word
    assert best_dictionary_word(model, tok, ctx, words, beams=1)[0].lower() in words
    assert best_dictionary_word(model, tok, ctx, frozenset({"ghost"}), exclude={"ghost"}) is None
