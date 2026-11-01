"""LLM access. Gemini API free tier only — every call goes through the rate limiter and daily budget."""
from __future__ import annotations

import json
import re
import time
from typing import Callable, Protocol

from .config import Settings
from .ratelimit import BudgetExceeded, RateLimiter, UsageTracker, estimate_tokens

MAX_RETRIES = 3
MAX_WAIT_SECONDS = 90  # longer waits mean a daily limit was hit: stop instead of hanging


class ModelUnavailable(RuntimeError):
    """Model overloaded (503) or its free quota is used up — try the next fallback model."""


class LLMClient(Protocol):
    def chat(self, messages: list[dict], *, model: str, json_mode: bool = False,
             max_tokens: int | None = None) -> str: ...


def _retry_delay(error) -> float | None:
    """Seconds Google asks us to wait, from the RetryInfo detail ("34s")."""
    m = re.search(r'"retryDelay":\s*"([\d.]+)s"', json.dumps(getattr(error, "details", None) or {}))
    return float(m.group(1)) if m else None


def _is_daily_quota(error) -> bool:
    text = json.dumps(getattr(error, "details", None) or {}) + str(getattr(error, "message", ""))
    return bool(re.search(r"per ?day|PerDay", text, re.I))


class GeminiClient:
    def __init__(self, settings: Settings, tracker: UsageTracker | None = None,
                 limiter: RateLimiter | None = None, client=None,
                 sleep: Callable[[float], None] = time.sleep):
        self.settings = settings
        self.tracker = tracker or UsageTracker(settings.daily_token_budget, settings.daily_request_budget)
        self._shared_limiter = limiter  # tests inject one; otherwise one per model (free limits are per model)
        self._limiters: dict[str, RateLimiter] = {}
        self.sleep = sleep
        if client is None:
            from google import genai
            from google.genai import types

            # attempts=1: no hidden SDK retries — retries are handled here so they respect our limits
            client = genai.Client(api_key=settings.require_gemini_key(),
                                  http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1)))
        self.client = client

    def _limiter(self, model: str) -> RateLimiter:
        if self._shared_limiter:
            return self._shared_limiter
        if model not in self._limiters:
            self._limiters[model] = RateLimiter(self.settings.rpm, self.settings.tpm, sleep=self.sleep)
        return self._limiters[model]

    def chat(self, messages: list[dict], *, model: str, json_mode: bool = False,
             max_tokens: int | None = None) -> str:
        """Try `model`, then the free fallback models, until one answers."""
        candidates = list(dict.fromkeys([model, *self.settings.fallback_models]))
        problems = []
        for i, name in enumerate(candidates):
            try:
                return self._chat_one(messages, name, json_mode, max_tokens)
            except ModelUnavailable as e:
                problems.append(f"{name}: {e}")
                if i + 1 < len(candidates):
                    print(f"  … {name} unavailable, switching to {candidates[i + 1]}")
        details = "\n  ".join(problems)
        raise BudgetExceeded(f"No free Gemini model available right now:\n  {details}\n"
                             "Progress is saved - run the same command again later.")

    def _chat_one(self, messages: list[dict], model: str, json_mode: bool, max_tokens: int | None) -> str:
        from google.genai import errors, types

        max_tokens = max_tokens or self.settings.max_output_tokens
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system") or None
        prompt = "\n\n".join(m["content"] for m in messages if m["role"] != "system")
        estimate = estimate_tokens((system or "") + prompt) + max_tokens
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0,
            max_output_tokens=max_tokens,
            response_mime_type="application/json" if json_mode else None,
            # Gemini 3 always thinks a bit; keep it short so it doesn't eat the output budget
            thinking_config=types.ThinkingConfig(thinking_level=self.settings.thinking_level),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),  # no tools used
        )

        for attempt in range(MAX_RETRIES + 1):
            try:
                self.tracker.check(model, estimate)
            except BudgetExceeded as e:
                raise ModelUnavailable(f"our daily budget reached ({e})") from e
            self._limiter(model).acquire(estimate)
            try:
                resp = self.client.models.generate_content(model=model, contents=prompt, config=config)
            except errors.ClientError as e:
                if e.code != 429:
                    raise
                wait = _retry_delay(e) or min(5 * 2 ** attempt, 60)
                if _is_daily_quota(e):
                    self.tracker.mark_exhausted(model)
                if _is_daily_quota(e) or wait > MAX_WAIT_SECONDS or attempt == MAX_RETRIES:
                    raise ModelUnavailable(
                        f"free-tier quota hit (retry after ~{int(wait)}s; daily quotas reset at midnight Pacific time)"
                    ) from e
                print(f"  … Gemini rate limit, waiting {wait:.0f}s")
                self.sleep(wait)
                continue
            except errors.ServerError as e:
                if attempt == MAX_RETRIES:
                    raise ModelUnavailable(f"server error {e.code} ({e.status}) after {attempt + 1} tries") from e
                self.sleep(5 * 2 ** attempt)
                continue
            usage = getattr(resp, "usage_metadata", None)
            used = usage.total_token_count if usage and usage.total_token_count else estimate
            self.tracker.add(model, used)
            self._limiter(model).correct_last(used)
            content = resp.text or ""
            if not content.strip():
                raise RuntimeError(f"{model} returned an empty answer (output limit reached?) - "
                                   "raise MAX_OUTPUT_TOKENS in .env")
            return content
        raise RuntimeError("unreachable")
