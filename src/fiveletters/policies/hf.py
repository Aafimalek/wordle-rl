from __future__ import annotations

from ..game import Game
from ..prompts import DEFAULT_FORMAT, PromptFormat, render_prompt
from .base import Policy


class HFPolicy(Policy):
    """A local Hugging Face causal LM, greedy by default. Batched across games."""

    def __init__(self, model_name: str, fmt: PromptFormat = DEFAULT_FORMAT, *, batch_size: int = 64,
                 max_new_tokens: int = 96, device: str | None = None, model=None, tokenizer=None,
                 do_sample: bool = False, temperature: float = 1.0, template_kwargs: dict | None = None,
                 precision: str = "fp32", constrain: str | None = None, constrain_mode: str = "greedy",
                 beams: int = 16):
        from ..lm import load_model_and_tokenizer, pick_device

        self.device = device or pick_device()
        import torch

        self.precision = precision
        self.constrain = constrain  # None, or 'wordle' | 'shared': dictionary-constrained decoding
        if constrain_mode not in ("greedy", "word"):
            raise ValueError("constrain_mode must be 'greedy' (token by token) or 'word' (best whole word)")
        if constrain_mode == "word" and not (fmt.spaced or fmt.guess_word):
            raise ValueError("word-level constrained decoding needs spaced letters or whole-word guesses")
        self.constrain_mode, self.beams = constrain_mode, beams
        if constrain:
            from ..words import guess_vocab

            self._constrain_words = frozenset(guess_vocab(constrain))
        else:
            self._constrain_words = None
        if model is None or tokenizer is None:
            dtype = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
            model, tokenizer = load_model_and_tokenizer(model_name, self.device, for_training=False, dtype=dtype)
        self.model, self.tok = model, tokenizer
        self.fmt, self.batch_size, self.max_new_tokens = fmt, batch_size, max_new_tokens
        self.do_sample, self.temperature = do_sample, temperature
        self.template_kwargs = template_kwargs or {}
        self.model_name = model_name
        parts = model_name.replace("\\", "/").rstrip("/").split("/")
        # checkpoint folders end in best/final/latest: keep the run name so evals of different runs never collide
        self.name = "hf-" + ("-".join(parts[-2:]) if parts[-1] in ("best", "final", "latest") and len(parts) > 1 else parts[-1])

    def describe(self) -> dict:
        return {"name": self.name, "model": self.model_name, "do_sample": self.do_sample,
                "max_new_tokens": self.max_new_tokens, "template_kwargs": self.template_kwargs,
                **({"precision": self.precision} if self.precision != "fp32" else {}),
                **({"constrained_decoding": self.constrain} if self.constrain else {}),
                **({"constrain_mode": "word", "beams": self.beams} if self.constrain and self.constrain_mode == "word"
                   else {}),
                "format": self.fmt.to_dict()}

    def chat(self, conversations: list[list[dict]]) -> list[str]:
        """One reply per conversation (system/user/assistant message lists)."""
        from ..lm import generate

        outs = generate(self.model, self.tok, conversations, max_new_tokens=self.max_new_tokens,
                        do_sample=self.do_sample, temperature=self.temperature, batch_size=self.batch_size,
                        template_kwargs=self.template_kwargs, constrain_words=self._constrain_words,
                        word_search=self._word_search(None, len(conversations)))
        return [o.text for o in outs]

    def act(self, games: list[Game]) -> list[str]:
        from ..lm import generate

        prompts = [render_prompt(g, self.fmt) for g in games]
        outs = generate(self.model, self.tok, prompts, max_new_tokens=self.max_new_tokens,
                        do_sample=self.do_sample, temperature=self.temperature, batch_size=self.batch_size,
                        template_kwargs=self.template_kwargs, constrain_words=self._constrain_words,
                        word_search=self._word_search(games, len(prompts)))
        return [o.text for o in outs]

    def _word_search(self, games, n) -> dict | None:
        if not self.constrain or self.constrain_mode != "word":
            return None
        # never replay a word already tried in this game (a rule about the game, not about the clues)
        exclude = [{t.guess for t in g.turns if t.guess} for g in games] if games else [set()] * n
        return {"beams": self.beams, "exclude": exclude, "guess_word": self.fmt.guess_word}
