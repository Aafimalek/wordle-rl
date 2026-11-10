"""Dictionary-constrained decoding: inside <guess>...</guess> the model may only type letters that keep it
on the path to a real word, then must close the tag. Everything outside the tag is untouched.

Real Wordle also rejects non-words (you retype without losing a turn), so this is a fair, clearly labeled
inference setting. Results with it are always reported separately from unconstrained decoding.
"""

from __future__ import annotations

import re
from functools import lru_cache

import torch

_LETTERS = re.compile(r"^ ?[A-Z]{1,5}$")
_TABLES: dict[str, tuple[list[tuple[int, str]], list[int]]] = {}


def _token_tables(tok) -> tuple[list[tuple[int, str]], list[int]]:
    """(letter tokens as (id, LETTERS), tokens that may start the closing tag). Cached per tokenizer."""
    key = f"{getattr(tok, 'name_or_path', '')}:{len(tok)}"
    if key not in _TABLES:
        letter_tokens, close_ids = [], []
        for i in range(len(tok)):
            s = tok.decode([i])
            if _LETTERS.match(s):
                letter_tokens.append((i, s.strip()))
            elif s.strip() in ("<", "</", "</guess", "</guess>"):
                close_ids.append(i)
        _TABLES[key] = (letter_tokens, close_ids)
    return _TABLES[key]


@lru_cache(maxsize=8)
def word_prefixes(words: frozenset[str]) -> frozenset[str]:
    return frozenset(w[:k].upper() for w in words for k in range(1, 6))


class GuessConstraint:
    """A transformers LogitsProcessor. `prompt_len` is the (left-padded) prompt length of the batch."""

    def __init__(self, tok, words: frozenset[str], prompt_len: int):
        self.tok = tok
        self.prefixes = word_prefixes(words)
        self.letter_tokens, self.close_ids = _token_tables(tok)
        self.prompt_len = prompt_len

    def allowed(self, generated_text: str) -> list[int] | None:
        """Token ids allowed next, or None for no constraint (outside a guess tag)."""
        start = generated_text.rfind("<guess>")
        if start < 0:
            return None
        body = generated_text[start + len("<guess>"):]
        if "<" in body:  # closing tag started: done
            return None
        letters = re.sub(r"[^A-Za-z]", "", body).upper()
        if len(letters) >= 5:
            return self.close_ids or None
        if letters and letters not in self.prefixes:
            return None  # typed before the constraint could act; the game judges it
        ok = [i for i, lt in self.letter_tokens if len(letters) + len(lt) <= 5 and (letters + lt) in self.prefixes]
        return ok or None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for row in range(input_ids.shape[0]):
            text = self.tok.decode(input_ids[row, self.prompt_len:].tolist(), skip_special_tokens=True)
            ok = self.allowed(text)
            if ok is None:
                continue
            keep = scores[row, ok].clone()
            scores[row, :] = float("-inf")
            scores[row, ok] = keep
        return scores


# ---------------------------------------------------------------------------------------------------------------
# Word-level constrained decoding (round v5+). Token-greedy constraining commits to the most likely allowed first
# letters and is then forced into whatever dictionary word continues them (ADLED -> ADDED), even when the model
# would rate another real word far higher as a whole. Here the model scores whole words instead: a beam search over
# the dictionary trie keeps the `beams` best letter prefixes by the model's own summed log-probability and returns
# the best complete word. The code only enforces "a real word not played yet in this game"; it never looks at the
# clues, so the model still has to work out which word fits them.
# ---------------------------------------------------------------------------------------------------------------

_LETTER_IDS: dict[str, tuple[list[int], list[int]]] = {}


def spaced_letter_ids(tok) -> tuple[list[int], list[int]] | None:
    """Token ids of 'A'..'Z' (first letter, right after <guess>) and ' A'..' Z' (later letters), or None if a
    letter is not a single token in this tokenizer (word-level search then falls back to the plain output)."""
    key = f"{getattr(tok, 'name_or_path', '')}:{len(tok)}"
    if key not in _LETTER_IDS:
        first, rest = [], []
        for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            a, b = tok(c, add_special_tokens=False)["input_ids"], tok(" " + c, add_special_tokens=False)["input_ids"]
            if len(a) != 1 or len(b) != 1:
                _LETTER_IDS[key] = None  # type: ignore[assignment]
                return None
            first.append(a[0])
            rest.append(b[0])
        _LETTER_IDS[key] = (first, rest)
    return _LETTER_IDS[key]


@torch.no_grad()
def best_dictionary_word(model, tok, context_ids: list[int], words: frozenset[str], exclude=frozenset(),
                         beams: int = 16) -> tuple[str, float] | None:
    """The real word the model rates highest after `context_ids` (which ends with '<guess>'), by beam search over
    the dictionary trie. Returns (word, summed log-prob) or None if no allowed word exists."""
    ids = spaced_letter_ids(tok)
    if ids is None:
        return None
    first, rest = ids
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    upper = frozenset(w.upper() for w in words)
    prefixes = word_prefixes(upper)
    banned = frozenset(w.upper() for w in exclude)
    device = next(model.parameters()).device
    out = model(input_ids=torch.tensor([context_ids], device=device), use_cache=True)
    past, logits = out.past_key_values, out.logits[:, -1]
    beam: list[tuple[str, float]] = [("", 0.0)]
    for depth in range(5):
        lp = torch.log_softmax(logits.float(), dim=-1)[:, first if depth == 0 else rest].cpu()  # [rows, 26]
        cands = []
        for r, (pre, score) in enumerate(beam):
            for j, c in enumerate(letters):
                p = pre + c
                ok = (p in upper and p not in banned) if depth == 4 else p in prefixes
                if ok:
                    cands.append((score + float(lp[r, j]), r, p, j))
        if not cands:
            return None
        cands.sort(key=lambda x: -x[0])
        cands = cands[:beams]
        if depth == 4:
            return cands[0][2], cands[0][0]
        parents = torch.tensor([c[1] for c in cands], device=device)
        past.reorder_cache(parents)
        step = torch.tensor([[(first if depth == 0 else rest)[c[3]]] for c in cands], device=device)
        out = model(input_ids=step, past_key_values=past, use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1]
        beam = [(c[2], c[0]) for c in cands]
    return None


_TOKEN_TRIES: dict = {}


def _token_trie(tok, words: frozenset[str]):
    """children[prefix] = next token ids; end[prefix] = word. Prefixes are tuples of token ids."""
    key = (getattr(tok, "name_or_path", ""), len(tok), hash(words))
    if key not in _TOKEN_TRIES:
        children: dict[tuple, set] = {}
        end: dict[tuple, str] = {}
        for w in words:
            ids = tuple(tok(w.lower(), add_special_tokens=False)["input_ids"])
            for k in range(len(ids)):
                children.setdefault(ids[:k], set()).add(ids[k])
            end[ids] = w.lower()
        close = tok("</guess>", add_special_tokens=False)["input_ids"][0]
        _TOKEN_TRIES[key] = (children, end, close)
    return _TOKEN_TRIES[key]


@torch.no_grad()
def best_dictionary_word_tokens(model, tok, context_ids: list[int], words: frozenset[str], exclude=frozenset(),
                                beams: int = 16) -> tuple[str, float] | None:
    """Whole-word format: the real word (1-3 tokens) the model rates highest after `context_ids` (ending with
    '<guess>'), including the probability of then closing the tag. Beam search over the token trie; partial
    prefixes that already score below the best finished word are dropped (scores only fall)."""
    children, end, close = _token_trie(tok, words)
    banned = frozenset(w.lower() for w in exclude)
    device = next(model.parameters()).device
    out = model(input_ids=torch.tensor([context_ids], device=device), use_cache=True)
    past, logits = out.past_key_values, out.logits[:, -1]
    beam: list[tuple[tuple, float]] = [((), 0.0)]
    best: tuple[str, float] | None = None
    for _ in range(8):
        lp = torch.log_softmax(logits.float(), dim=-1).cpu()
        cands = []
        for r, (pre, score) in enumerate(beam):
            if pre in end and end[pre] not in banned:  # this row's next token may close the tag
                s = score + float(lp[r, close])
                if best is None or s > best[1]:
                    best = (end[pre], s)
            for t in children.get(pre, ()):
                s = score + float(lp[r, t])
                if best is None or s > best[1]:
                    cands.append((s, r, pre + (t,), t))
        if not cands:
            break
        cands.sort(key=lambda x: -x[0])
        cands = cands[:beams]
        past.reorder_cache(torch.tensor([c[1] for c in cands], device=device))
        out = model(input_ids=torch.tensor([[c[3]] for c in cands], device=device), past_key_values=past,
                    use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1]
        beam = [(c[2], c[0]) for c in cands]
    return best
