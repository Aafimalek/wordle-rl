"""Model loading, batched generation and completion log-probs, shared by eval, SFT and GRPO.

Free-GPU details handled here:

* Weights always stay float32. Matmuls autocast to bfloat16 where the GPU has it.
  T4s (Kaggle, Colab) lack bfloat16, but their float16 tensor cores are about 8x faster
  than float32. Gemma 3 is known to overflow in float16, so training first checks
  float16 against float32 on real batches (``choose_precision``), uses it only if they
  agree, and falls back to float32 if non-finite losses show up later.
* Gemma 3's vocabulary has 262k entries, so full logits are huge. Training computes
  logits only for completion positions (``logits_to_keep``), with sequences left-padded
  so every completion ends at the same position.
* Left padding needs explicit position ids, or padded rows get shifted positions that
  do not match what ``generate`` used.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch

from .prompts import as_messages


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def native_bf16() -> bool:
    """True only on GPUs with bfloat16 tensor cores (Ampere, sm80+). Newer PyTorch reports bf16 as
    'supported' on T4 too, but there it is emulated and many times slower than float32."""
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] >= 8 and torch.cuda.is_bf16_supported()


_AMP: dict = {"set": False, "dtype": None}


def set_amp_dtype(dtype) -> None:
    """Fix the autocast dtype for this process (None = plain float32)."""
    _AMP.update(set=True, dtype=dtype)


def amp_dtype(device: str):
    """The autocast dtype in use. Default: bfloat16 where supported, else float32 (None).
    Evaluation keeps this default, so reported numbers never depend on float16."""
    if _AMP["set"]:
        return _AMP["dtype"]
    if device.startswith("cuda") and native_bf16():
        return torch.bfloat16
    return None


def autocast(device: str):
    dt = amp_dtype(device)
    if dt is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=dt)


@torch.no_grad()
def choose_precision(model, pairs, pad_id: int, requested: str = "auto"):
    """Pick the training autocast dtype. 'fp32' | 'bf16' | 'fp16' | 'auto'.

    auto: bfloat16 if the GPU has it. Otherwise (T4) float16, but only if completion
    log-probs under float16 are finite and within 0.02 nats of float32 on real batches.
    On CPU, auto means float32."""
    device = str(next(model.parameters()).device)
    cuda = device.startswith("cuda")
    if requested == "fp32":
        return None
    if requested == "bf16":
        return torch.bfloat16
    if requested == "fp16":
        return torch.float16
    if requested != "auto":
        raise ValueError(f"unknown precision {requested!r}")
    if not cuda:
        return None
    if native_bf16():
        return torch.bfloat16
    was_training = model.training
    model.eval()
    ids, attn, pos, mask, K = build_batch(pairs[:8], pad_id, device)
    saved = dict(_AMP)
    try:
        set_amp_dtype(None)
        ref = completion_logprobs(model, ids, attn, pos, K)
        set_amp_dtype(torch.float16)
        half = completion_logprobs(model, ids, attn, pos, K)
    finally:
        _AMP.update(saved)
        model.train(was_training)
    m = mask.bool()
    finite = bool(torch.isfinite(half[m]).all())
    diff = float((half[m] - ref[m]).abs().mean()) if finite else float("inf")
    print(f"[precision] float16 check on {len(pairs[:8])} sequences: finite={finite}, mean |dlogp|={diff:.4f}")
    return torch.float16 if finite and diff < 0.02 else None


class MixedPrecision:
    """Loss scaling for float16, with automatic fallback to float32 after repeated non-finite losses."""

    def __init__(self, dtype, device: str, patience: int = 3):
        self.dtype = dtype
        self.patience = patience
        self.bad = 0
        self.fell_back = False
        self.scaler = torch.amp.GradScaler("cuda" if device.startswith("cuda") else "cpu") if dtype == torch.float16 else None
        set_amp_dtype(dtype)

    @property
    def name(self) -> str:
        return {None: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}[self.dtype]

    def backward(self, loss) -> None:
        (self.scaler.scale(loss) if self.scaler else loss).backward()

    def step(self, model, opt, clip: float) -> float:
        if self.scaler:
            self.scaler.unscale_(opt)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip).item()
        if self.scaler:
            self.scaler.step(opt)  # skipped automatically when gradients overflowed
            self.scaler.update()
        else:
            opt.step()
        return norm

    def check(self, loss_value: float) -> None:
        import math

        if math.isfinite(loss_value):
            self.bad = 0
            return
        self.bad += 1
        if self.dtype is not None and self.bad >= self.patience:
            print(f"[precision] {self.bad} non-finite losses in a row under {self.name}; switching to fp32 for the rest of the run")
            self.dtype, self.scaler, self.fell_back, self.bad = None, None, True, 0
            set_amp_dtype(None)

    def state_dict(self) -> dict:
        return {"dtype": self.name, "fell_back": self.fell_back,
                "scaler": self.scaler.state_dict() if self.scaler else None}

    def load_state_dict(self, st: dict | None) -> None:
        if not st:
            return
        if st.get("fell_back"):  # a previous session found float16 unstable: stay on float32
            self.dtype, self.scaler, self.fell_back = None, None, True
            set_amp_dtype(None)
        elif self.scaler and st.get("scaler"):
            self.scaler.load_state_dict(st["scaler"])


def load_model_and_tokenizer(name_or_path: str, device: str, *, for_training: bool,
                             attn_implementation: str | None = None, dtype=None):
    """Weights load in float32 unless `dtype` is given (inference only: e.g. float16 for a 4B baseline on a T4)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name_or_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    import transformers

    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    kwargs = {"dtype" if (major, minor) >= (4, 56) else "torch_dtype": dtype or torch.float32}
    if attn_implementation is None and "gemma" in name_or_path.lower() and for_training:
        attn_implementation = "eager"  # recommended for training Gemma 3
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(name_or_path, **kwargs).to(device)
    model.train(for_training)
    if not for_training:
        model.eval()
    return model, tok


def chat_text(tok, prompt, **template_kwargs) -> str:
    """A prompt string (one user turn) or a full message list, rendered with the chat template.
    template_kwargs go to the template, e.g. enable_thinking=False for Qwen3 baselines."""
    messages = as_messages(prompt) if isinstance(prompt, str) else prompt
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **template_kwargs)


def completion_suffix(tok) -> str:
    """Text the chat template puts after an assistant message (e.g. Gemma's '<end_of_turn>\\n')."""
    probe = "PROBE_COMPLETION_7Q"
    prompt = chat_text(tok, "x")
    full = tok.apply_chat_template(as_messages("x") + [{"role": "assistant", "content": probe}], tokenize=False)
    if full.startswith(prompt) and probe in full:
        return full[len(prompt):].split(probe, 1)[1]
    return tok.eos_token or ""


def stop_token_ids(model, tok) -> list[int]:
    ids: set[int] = set()
    gen_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if isinstance(gen_eos, int):
        ids.add(gen_eos)
    elif gen_eos:
        ids.update(gen_eos)
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    for t in ("<end_of_turn>", "<|im_end|>", "<|eot_id|>"):
        tid = tok.convert_tokens_to_ids(t)
        if isinstance(tid, int) and tid != tok.unk_token_id and tid >= 0:
            ids.add(tid)
    return sorted(ids)


@dataclass
class Generation:
    prompt_ids: list[int]
    completion_ids: list[int]  # includes the stop token when one was produced
    text: str


@torch.no_grad()
def generate(model, tok, prompts: list, *, max_new_tokens: int = 96, do_sample: bool = False,
             temperature: float = 1.0, top_p: float = 1.0, batch_size: int = 64,
             template_kwargs: dict | None = None, constrain_words: frozenset | None = None,
             word_search: dict | None = None) -> list[Generation]:
    """word_search = {"beams": int, "exclude": [set of words per prompt]}: with constrain_words, choose the guess as
    the best whole dictionary word (see constrain.best_dictionary_word) instead of token-greedy constraining."""
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    stops = stop_token_ids(model, tok)
    pad = tok.pad_token_id
    encoded = [tok(chat_text(tok, p, **(template_kwargs or {})), add_special_tokens=False)["input_ids"] for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: len(encoded[i]))
    results: list[Generation | None] = [None] * len(prompts)
    bs = min(batch_size, _GEN_BATCH_CAP.get(id(model), batch_size))
    s = 0
    while s < len(order):
        idx = order[s : s + bs]
        try:
            _generate_chunk(model, tok, idx, encoded, results, stops, pad, device, max_new_tokens, do_sample,
                            temperature, top_p, None if word_search else constrain_words)
        except torch.cuda.OutOfMemoryError:
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            _GEN_BATCH_CAP[id(model)] = bs  # remember for the rest of this run
            torch.cuda.empty_cache()
            print(f"[generate] out of GPU memory; retrying with batch size {bs}", flush=True)
            continue
        s += len(idx)
    if word_search and constrain_words:
        _choose_whole_words(model, tok, encoded, results, stops, constrain_words, word_search)
    model.train(was_training)
    return results  # type: ignore[return-value]


def _choose_whole_words(model, tok, encoded, results, stops, words, word_search) -> None:
    """Replace each generated guess by the best whole dictionary word the model can produce after its own
    '<state>...<guess>' prefix. A reply without '<guess>' is left as is (the game judges it)."""
    from .constrain import best_dictionary_word, best_dictionary_word_tokens

    beams = int(word_search.get("beams", 16))
    whole = bool(word_search.get("guess_word"))
    search = best_dictionary_word_tokens if whole else best_dictionary_word
    excludes = word_search.get("exclude") or [()] * len(results)
    end = stops[0] if stops else None
    device = str(next(model.parameters()).device)
    for i, g in enumerate(results):
        k = g.text.find("<guess>")
        if k < 0:
            continue
        prefix = g.text[: k + len("<guess>")]
        ctx = encoded[i] + tok(prefix, add_special_tokens=False)["input_ids"]
        with autocast(device):
            best = search(model, tok, ctx, words, frozenset(excludes[i]), beams=beams)
        if best is None:
            continue
        text = prefix + (best[0].lower() if whole else " ".join(best[0].upper())) + "</guess>"
        comp = tok(text, add_special_tokens=False)["input_ids"] + ([end] if end is not None else [])
        results[i] = Generation(g.prompt_ids, comp, text)


_GEN_BATCH_CAP: dict[int, int] = {}


def _generate_chunk(model, tok, idx, encoded, results, stops, pad, device, max_new_tokens, do_sample,
                    temperature, top_p, constrain_words=None) -> None:
    seqs = [encoded[i] for i in idx]
    L = max(len(x) for x in seqs)
    input_ids = torch.full((len(seqs), L), pad, dtype=torch.long)
    attn = torch.zeros((len(seqs), L), dtype=torch.long)
    for r, x in enumerate(seqs):
        input_ids[r, L - len(x):] = torch.tensor(x)
        attn[r, L - len(x):] = 1
    kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample, "eos_token_id": stops, "pad_token_id": pad}
    if do_sample:
        kwargs.update(temperature=temperature, top_p=top_p, top_k=0)
    if constrain_words:
        from transformers import LogitsProcessorList

        from .constrain import GuessConstraint

        kwargs["logits_processor"] = LogitsProcessorList([GuessConstraint(tok, constrain_words, L)])
    with autocast(str(device)):
        out = model.generate(input_ids=input_ids.to(device), attention_mask=attn.to(device), **kwargs)
    out = out[:, L:].tolist()
    for r, i in enumerate(idx):
        comp = out[r]
        cut = len(comp)
        for j, t in enumerate(comp):
            if t in stops:
                cut = j + 1
                break
        comp = comp[:cut]
        text = tok.decode([t for t in comp if t not in stops], skip_special_tokens=True)
        results[i] = Generation(encoded[i], comp, text)

def build_batch(pairs: list[tuple[list[int], list[int]]], pad_id: int, device):
    """Left-pad prompt+completion so every completion ends at the last position."""
    K = max(len(c) for _, c in pairs)
    T = max(len(p) + len(c) for p, c in pairs)
    B = len(pairs)
    input_ids = torch.full((B, T), pad_id, dtype=torch.long)
    attn = torch.zeros((B, T), dtype=torch.long)
    comp_mask = torch.zeros((B, K), dtype=torch.float32)
    for r, (p, c) in enumerate(pairs):
        seq = p + c
        input_ids[r, T - len(seq):] = torch.tensor(seq)
        attn[r, T - len(seq):] = 1
        comp_mask[r, K - len(c):] = 1.0
    pos = (attn.cumsum(-1) - 1).clamp(min=0)
    return input_ids.to(device), attn.to(device), pos.to(device), comp_mask.to(device), K


def completion_logprobs(model, input_ids, attn, pos, K: int) -> torch.Tensor:
    """Log-prob of each of the last K tokens given everything before it. Shape (B, K)."""
    device = str(input_ids.device)
    with autocast(device):
        try:
            out = model(input_ids=input_ids, attention_mask=attn, position_ids=pos, logits_to_keep=K + 1)
        except TypeError:  # older model classes without logits_to_keep
            out = model(input_ids=input_ids, attention_mask=attn, position_ids=pos)
    logits = out.logits[:, -(K + 1):-1, :].float()
    targets = input_ids[:, -K:]
    return torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
