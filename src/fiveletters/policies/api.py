"""Baselines through free, OpenAI-compatible chat APIs (Gemini, Groq, OpenRouter, Cerebras).

Built for free tiers: a requests-per-minute limiter, backoff that honors Retry-After,
and a clean ``QuotaExhausted`` when the daily cap is hit. The harness saves every
finished game, so the next run (tomorrow, when the cap resets) picks up where this
one stopped. Only the standard library is used.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from ..game import Game
from ..prompts import DEFAULT_FORMAT, PromptFormat, as_messages, render_prompt
from .base import Policy, QuotaExhausted

PROVIDERS = {
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "cerebras": ("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY"),
    "nvidia": ("https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY"),  # build.nvidia.com free credits
}
_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


class ChatAPIPolicy(Policy):
    def __init__(self, provider: str, model: str, fmt: PromptFormat = DEFAULT_FORMAT, *,
                 base_url: str | None = None, api_key_env: str | None = None,
                 requests_per_minute: float = 10.0, max_tokens: int = 2048, temperature: float = 0.0,
                 max_retries: int = 30, give_up_after_s: float = 900.0, timeout_s: float = 300.0,
                 concurrency: int = 8):
        if provider not in PROVIDERS and not base_url:
            raise ValueError(f"unknown provider {provider!r}; pass base_url for a custom endpoint")
        default_url, default_env = PROVIDERS.get(provider, (None, None))
        self.base_url = (base_url or default_url).rstrip("/")
        env = api_key_env or default_env
        self.api_key = os.environ.get(env or "", "")
        if not self.api_key:
            raise RuntimeError(f"set the {env} environment variable (free key from the provider's console)")
        self.provider, self.model, self.fmt = provider, model, fmt
        self.min_interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self.max_tokens, self.temperature = max_tokens, temperature
        self.max_retries, self.give_up_after_s, self.timeout_s = max_retries, give_up_after_s, timeout_s
        self._last = 0.0
        self._lock = threading.Lock()  # the rate limit holds across all worker threads
        self.concurrency = max(1, concurrency)
        self.retry_counts: dict[str, int] = {}
        self.name = f"api-{provider}-{model}"
        self.requests_made = 0

    def describe(self) -> dict:
        return {"name": self.name, "provider": self.provider, "model": self.model,
                "temperature": self.temperature, "max_tokens": self.max_tokens, "format": self.fmt.to_dict()}

    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}",
                     "User-Agent": "fiveletters-eval"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode())

    def _complete(self, prompt) -> str:
        messages = as_messages(prompt) if isinstance(prompt, str) else prompt
        payload = {"model": self.model, "messages": messages,
                   "temperature": self.temperature, "max_tokens": self.max_tokens}
        started = time.monotonic()
        delay = 5.0
        for attempt in range(self.max_retries + 1):
            with self._lock:  # reserve the next send slot, then wait for it outside the lock
                slot = max(time.monotonic(), self._last + self.min_interval)
                self._last = slot
            if slot > time.monotonic():
                time.sleep(slot - time.monotonic())
            try:
                self.requests_made += 1
                data = self._post(payload)
                msg = (data.get("choices") or [{}])[0].get("message") or {}
                return msg.get("content") or ""
            except urllib.error.HTTPError as e:
                if e.code in (401, 403):
                    raise RuntimeError(f"{self.provider} rejected the API key (HTTP {e.code})") from e
                if e.code not in _RETRY_STATUS:
                    body = e.read().decode(errors="replace")[:300]
                    raise RuntimeError(f"{self.provider} HTTP {e.code}: {body}") from e
                retry_after = e.headers.get("Retry-After") if e.headers else None
                sleep_s = float(retry_after) if retry_after and retry_after.replace(".", "").isdigit() else delay
                why = f"HTTP {e.code}"
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                sleep_s = delay
                why = type(e).__name__
            self._note_retry(why)
            if attempt == self.max_retries or time.monotonic() - started + sleep_s > self.give_up_after_s:
                raise QuotaExhausted(
                    f"{self.provider} kept refusing requests for {time.monotonic() - started:.0f}s. "
                    "Finished games are saved; rerun the same command later to continue."
                )
            time.sleep(sleep_s)
            delay = min(delay * 2, 120.0)
        raise QuotaExhausted("retries exhausted")

    def _map(self, items) -> list[str]:
        if self.concurrency == 1 or len(items) == 1:
            return [self._complete(x) for x in items]
        with ThreadPoolExecutor(max_workers=min(self.concurrency, len(items))) as ex:
            return list(ex.map(self._complete, items))  # order is preserved; the first error propagates

    def chat(self, conversations: list[list[dict]]) -> list[str]:
        return self._map(conversations)

    def _note_retry(self, why: str) -> None:
        with self._lock:
            self.retry_counts[why] = self.retry_counts.get(why, 0) + 1
            n = self.retry_counts[why]
        if n in (1, 10, 100) or n % 500 == 0:
            print(f"[api] {self.provider} retrying after {why} (seen {n}x)", flush=True)

    def act(self, games: list[Game]) -> list[str]:
        return self._map([render_prompt(g, self.fmt) for g in games])
