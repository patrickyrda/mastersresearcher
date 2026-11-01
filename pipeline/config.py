"""Settings loaded from .env (the only place secrets live)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INDEX_DIR = DATA_DIR / "index"
EXTRACTED_DIR = DATA_DIR / "extracted"
USAGE_FILE = DATA_DIR / "usage.json"
PROGRAMS_FILE = ROOT / "programs.yaml"
MANUAL_DIR = ROOT / "manual"  # pages saved by hand for sites that block crawlers

load_dotenv(ROOT / ".env")


class MissingKeyError(RuntimeError):
    pass


def _int(name: str, default: int) -> int:
    return int(os.getenv(name) or default)


def _float(name: str, default: float) -> float:
    return float(os.getenv(name) or default)


@dataclass(frozen=True)
class Settings:
    # repr=False so the key never shows up if settings get printed or logged
    gemini_api_key: str = field(default="", repr=False)
    extract_model: str = "gemini-3.8-flash"
    chat_model: str = "gemini-3.5-flash-lite"
    thinking_level: str = "low"
    # other free models tried when the main one is overloaded or out of free quota
    fallback_models: tuple[str, ...] = ("gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.1-flash-lite")
    rpm: int = 5
    tpm: int = 100_000
    daily_request_budget: int = 20
    daily_token_budget: int = 1_000_000
    max_output_tokens: int = 8000
    embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    crawl_max_pages: int = 20
    crawl_depth: int = 2
    crawl_delay: float = 1.0
    user_agent: str = "MasterResearchBot/0.1 (personal study research; polite crawler)"

    def require_gemini_key(self) -> str:
        if not self.gemini_api_key:
            raise MissingKeyError(
                "GEMINI_API_KEY not set - copy .env.example to .env and paste your free key "
                "from https://aistudio.google.com/apikey"
            )
        return self.gemini_api_key


def load_settings() -> Settings:
    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        extract_model=os.getenv("EXTRACT_MODEL") or Settings.extract_model,
        chat_model=os.getenv("CHAT_MODEL") or Settings.chat_model,
        thinking_level=os.getenv("THINKING_LEVEL") or Settings.thinking_level,
        fallback_models=tuple(m.strip() for m in os.getenv("FALLBACK_MODELS", "").split(",") if m.strip())
        or Settings.fallback_models,
        rpm=_int("GEMINI_RPM", Settings.rpm),
        tpm=_int("GEMINI_TPM", Settings.tpm),
        daily_request_budget=_int("DAILY_REQUEST_BUDGET", Settings.daily_request_budget),
        daily_token_budget=_int("DAILY_TOKEN_BUDGET", Settings.daily_token_budget),
        max_output_tokens=_int("MAX_OUTPUT_TOKENS", Settings.max_output_tokens),
        embed_model=os.getenv("EMBED_MODEL") or Settings.embed_model,
        crawl_max_pages=_int("CRAWL_MAX_PAGES", Settings.crawl_max_pages),
        crawl_depth=_int("CRAWL_DEPTH", Settings.crawl_depth),
        crawl_delay=_float("CRAWL_DELAY_SECONDS", Settings.crawl_delay),
    )


settings = load_settings()
