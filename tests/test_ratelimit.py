from datetime import date
from types import SimpleNamespace

import pytest
from google.genai import errors

from pipeline.config import Settings
from pipeline.llm import GeminiClient
from pipeline.ratelimit import BudgetExceeded, RateLimiter, UsageTracker


class FakeClock:
    def __init__(self):
        self.t = 0.0
        self.slept = []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


def _quota_error(retry_delay: str, quota_id: str = "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"):
    return errors.ClientError(429, {"error": {
        "code": 429, "message": "You exceeded your current quota.", "status": "RESOURCE_EXHAUSTED",
        "details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                     "violations": [{"quotaId": quota_id}]},
                    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay}]}})


def _ok(tokens=100, text='{"ok": true}'):
    return SimpleNamespace(usage_metadata=SimpleNamespace(total_token_count=tokens), text=text)


class FakeGemini:
    """Mimics client.models.generate_content; records the calls it receives."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, **kwargs):
        self.calls.append(kwargs)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def _client(fake, tmp_path, clock=None, **settings):
    clock = clock or FakeClock()
    return GeminiClient(Settings(**settings), tracker=UsageTracker(10**6, 10**6, path=tmp_path / "u.json"),
                        client=fake, limiter=RateLimiter(30, 100_000, clock=clock, sleep=clock.sleep),
                        sleep=clock.sleep)


MSGS = [{"role": "system", "content": "be precise"}, {"role": "user", "content": "hi"}]


def test_limiter_waits_when_rpm_exceeded():
    clock = FakeClock()
    lim = RateLimiter(rpm=2, tpm=10_000, clock=clock, sleep=clock.sleep)
    lim.acquire(10)
    lim.acquire(10)
    assert clock.slept == []
    lim.acquire(10)  # third request in the same minute must wait ~60s
    assert clock.t >= 60


def test_limiter_waits_when_tpm_exceeded():
    clock = FakeClock()
    lim = RateLimiter(rpm=100, tpm=1000, clock=clock, sleep=clock.sleep)
    lim.acquire(800)
    lim.acquire(800)
    assert clock.t >= 60


def test_daily_token_budget_stops_before_calling(tmp_path):
    tracker = UsageTracker(budget=1000, path=tmp_path / "u.json")
    tracker.add("m", 900)
    fake = FakeGemini([_ok()])
    client = GeminiClient(Settings(max_output_tokens=500, fallback_models=()), tracker=tracker, client=fake)
    with pytest.raises(BudgetExceeded):
        client.chat(MSGS, model="m")
    assert fake.calls == []  # nothing was sent


def test_daily_request_budget_stops_before_calling(tmp_path):
    tracker = UsageTracker(budget=10**6, request_budget=2, path=tmp_path / "u.json")
    tracker.add("m", 10)
    tracker.add("m", 10)
    fake = FakeGemini([_ok()])
    with pytest.raises(BudgetExceeded, match="request budget"):
        GeminiClient(Settings(fallback_models=()), tracker=tracker, client=fake).chat(MSGS, model="m")
    assert fake.calls == []


def test_usage_resets_each_day(tmp_path):
    day = [date(2026, 9, 26)]
    tracker = UsageTracker(budget=1000, path=tmp_path / "u.json", today=lambda: day[0])
    tracker.add("m", 700)
    assert tracker.used("m")["tokens"] == 700
    day[0] = date(2026, 9, 27)
    assert tracker.used("m")["tokens"] == 0


def test_budget_is_per_model(tmp_path):
    tracker = UsageTracker(budget=1000, path=tmp_path / "u.json")
    tracker.add("extract-model", 1000)
    with pytest.raises(BudgetExceeded):
        tracker.check("extract-model", 10)
    tracker.check("chat-model", 500)  # other model's quota untouched


def test_request_is_built_for_gemini(tmp_path):
    fake = FakeGemini([_ok(tokens=250)])
    client = _client(fake, tmp_path)
    assert client.chat(MSGS, model="gemini-x", json_mode=True, max_tokens=300) == '{"ok": true}'
    call = fake.calls[0]
    cfg = call["config"]
    assert call["model"] == "gemini-x" and call["contents"] == "hi"
    assert cfg.system_instruction == "be precise"
    assert cfg.temperature == 0 and cfg.max_output_tokens == 300
    assert cfg.response_mime_type == "application/json"
    assert cfg.thinking_config.thinking_level.value.lower() == "low"
    assert client.tracker.used("gemini-x") == {"tokens": 250, "requests": 1}


def test_429_short_delay_is_retried(tmp_path):
    clock = FakeClock()
    fake = FakeGemini([_quota_error("3s"), _ok()])
    client = _client(fake, tmp_path, clock)
    assert client.chat(MSGS, model="m") == '{"ok": true}'
    assert len(fake.calls) == 2 and 3.0 in clock.slept


def test_429_daily_quota_stops_run(tmp_path):
    clock = FakeClock()
    fake = FakeGemini([_quota_error("20s", quota_id="GenerateRequestsPerDayPerProjectPerModel-FreeTier")])
    with pytest.raises(BudgetExceeded, match="midnight Pacific"):
        _client(fake, tmp_path, clock, fallback_models=()).chat(MSGS, model="m")
    assert len(fake.calls) == 1 and clock.slept == []


def test_429_long_delay_stops_run(tmp_path):
    clock = FakeClock()
    fake = FakeGemini([_quota_error("3600s")])
    with pytest.raises(BudgetExceeded):
        _client(fake, tmp_path, clock, fallback_models=()).chat(MSGS, model="m")
    assert clock.slept == []


def _overloaded():
    return errors.ServerError(503, {"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}})


def test_overloaded_model_falls_back_to_next_free_model(tmp_path):
    clock = FakeClock()
    fake = FakeGemini([_overloaded()] * 4 + [_ok(tokens=42)])
    client = _client(fake, tmp_path, clock, fallback_models=("backup",))
    assert client.chat(MSGS, model="main") == '{"ok": true}'
    assert [c["model"] for c in fake.calls] == ["main"] * 4 + ["backup"]
    assert client.tracker.used("backup")["tokens"] == 42 and client.tracker.used("main")["requests"] == 0


def test_daily_quota_falls_back_then_stops_cleanly(tmp_path):
    daily = lambda: _quota_error("20s", quota_id="GenerateRequestsPerDayPerProjectPerModel-FreeTier")  # noqa: E731
    fake = FakeGemini([daily(), daily()])
    with pytest.raises(BudgetExceeded, match="No free Gemini model available"):
        _client(fake, tmp_path, fallback_models=("backup",)).chat(MSGS, model="main")
    assert [c["model"] for c in fake.calls] == ["main", "backup"]


def test_other_client_errors_are_not_retried(tmp_path):
    fake = FakeGemini([errors.ClientError(400, {"error": {"code": 400, "message": "bad model"}})])
    with pytest.raises(errors.ClientError):
        _client(fake, tmp_path).chat(MSGS, model="m")
    assert len(fake.calls) == 1


def test_empty_answer_fails_loudly(tmp_path):
    fake = FakeGemini([_ok(text=None)])
    with pytest.raises(RuntimeError, match="empty answer"):
        _client(fake, tmp_path).chat(MSGS, model="m")


def test_daily_quota_marks_model_exhausted_for_the_day(tmp_path):
    daily = _quota_error("20s", quota_id="GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    fake = FakeGemini([daily, _ok(), _ok()])
    client = _client(fake, tmp_path, fallback_models=("backup",))
    client.tracker.request_budget = 20
    client.chat(MSGS, model="main")
    client.chat(MSGS, model="main")  # main is known to be exhausted: goes straight to backup
    assert [c["model"] for c in fake.calls] == ["main", "backup", "backup"]
