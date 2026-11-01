"""Keep Gemini usage inside the free tier: requests/min, tokens/min, and daily request/token budgets.

Free quotas are per model, so the daily budgets are tracked per model too.
"""
from __future__ import annotations

import json
import time
from collections import deque
from datetime import date
from pathlib import Path
from typing import Callable

from .config import USAGE_FILE


def _pacific_today() -> date:
    """Gemini's daily quotas reset at midnight Pacific time, so 'today' is counted there."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    try:
        return datetime.now(ZoneInfo("America/Los_Angeles")).date()
    except Exception:  # no tz database installed: fall back to local date
        return date.today()


class BudgetExceeded(RuntimeError):
    """Daily budget (ours or Google's free-tier quota) reached — resume tomorrow."""


class UsageTracker:
    """Tokens and requests spent today per model, persisted to data/usage.json."""

    def __init__(self, budget: int, request_budget: int = 10**9, path: Path = USAGE_FILE,
                 today: Callable[[], date] = _pacific_today):
        self.budget = budget
        self.request_budget = request_budget
        self.path = path
        self.today = today

    def _load(self) -> dict:
        fresh = {"date": self.today().isoformat(), "models": {}}
        if not self.path.exists():
            return fresh
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return data if data.get("date") == fresh["date"] and "models" in data else fresh

    def used(self, model: str | None = None) -> dict:
        """Today's usage of one model ({tokens, requests}), or all models when model is None."""
        models = self._load()["models"]
        if model is None:
            return models
        return models.get(model, {"tokens": 0, "requests": 0})

    def check(self, model: str, estimated_tokens: int) -> None:
        used = self.used(model)
        if used["requests"] + 1 > self.request_budget:
            raise BudgetExceeded(
                f"Daily request budget for {model} reached ({used['requests']}/{self.request_budget}). "
                "Resume tomorrow (quotas reset at midnight Pacific time), or adjust DAILY_REQUEST_BUDGET in .env "
                "(stay under the limit shown at aistudio.google.com/rate-limit)."
            )
        spent = used["tokens"]
        if spent + estimated_tokens > self.budget:
            raise BudgetExceeded(
                f"Daily token budget for {model} reached ({spent:,}/{self.budget:,} used, next call needs "
                f"~{estimated_tokens:,}). Resume tomorrow, or raise DAILY_TOKEN_BUDGET in .env "
                "(stay under the free limit)."
            )

    def mark_exhausted(self, model: str) -> None:
        """Google reported the model's daily quota as used up: skip it for the rest of the day."""
        data = self._load()
        entry = data["models"].setdefault(model, {"tokens": 0, "requests": 0})
        entry["requests"] = max(entry["requests"], self.request_budget)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data), encoding="utf-8")

    def add(self, model: str, tokens: int) -> None:
        data = self._load()
        entry = data["models"].setdefault(model, {"tokens": 0, "requests": 0})
        entry["tokens"] += tokens
        entry["requests"] += 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data), encoding="utf-8")


class RateLimiter:
    """Sliding 60-second window over requests and tokens (one limiter per model)."""

    def __init__(self, rpm: int, tpm: int, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.rpm, self.tpm = rpm, tpm
        self.clock, self.sleep = clock, sleep
        self.events: deque[tuple[float, int]] = deque()

    def _prune(self, now: float) -> None:
        while self.events and now - self.events[0][0] >= 60:
            self.events.popleft()

    def acquire(self, tokens: int) -> None:
        tokens = min(tokens, self.tpm)  # a single oversized call must still be able to go
        while True:
            now = self.clock()
            self._prune(now)
            used = sum(t for _, t in self.events)
            if len(self.events) < self.rpm and used + tokens <= self.tpm:
                self.events.append((now, tokens))
                return
            wait = 60 - (now - self.events[0][0]) + 0.1
            self.sleep(max(wait, 0.1))

    def correct_last(self, actual_tokens: int) -> None:
        """Replace the estimate of the last call with the real token count."""
        if self.events:
            ts, _ = self.events.pop()
            self.events.append((ts, actual_tokens))


def estimate_tokens(text: str) -> int:
    return len(text) // 4 + 1
