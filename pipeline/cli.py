"""Command line: python -m pipeline.cli <command> --help"""
from __future__ import annotations

import sys
from typing import Optional

import typer

from .config import MissingKeyError, settings
from .discover import crawl, load_docs, save_docs
from .extract import extract_program, load_record, missing_fields
from .fetch import Fetcher
from .index import Index, build_index, get_embedder
from .ratelimit import BudgetExceeded, UsageTracker
from .schema import ProgramSeed, load_programs

# Windows consoles default to cp1252: don't crash on €, ≈, accents in page titles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

app = typer.Typer(add_completion=False, help="Finance master's research pipeline (Gemini free tier).")


def _select(program: Optional[str], all_: bool) -> list[ProgramSeed]:
    seeds = load_programs()
    if all_:
        return seeds
    if not program:
        raise typer.BadParameter("pass --program <id> or --all")
    chosen = [s for s in seeds if s.id == program]
    if not chosen:
        raise typer.BadParameter(f"unknown program '{program}'. Ids: {', '.join(s.id for s in seeds)}")
    return chosen


def crawl_program(seed: ProgramSeed, fetcher: Fetcher | None = None) -> int:
    print(f"== crawl {seed.id}")
    docs = crawl(seed, fetcher or Fetcher(settings), settings)
    save_docs(seed.id, docs)
    return len(docs)


def index_program(seed: ProgramSeed) -> int:
    n = build_index(seed.id, load_docs(seed.id), get_embedder(settings))
    print(f"== index {seed.id}: {n} chunks")
    return n


def extract_one(seed: ProgramSeed, force: bool = False, llm=None) -> None:
    from .llm import GeminiClient

    print(f"== extract {seed.id}")
    record = extract_program(seed, Index.load([seed.id]), get_embedder(settings), llm or GeminiClient(settings),
                             load_docs(seed.id), settings, force=force)
    gaps = missing_fields(record)
    print(f"   done — {len(gaps)} missing/unverified fields")


def run_program(seed: ProgramSeed, force: bool = False, fetcher: Fetcher | None = None) -> None:
    """Full pipeline for one program; skips steps whose output already exists unless force."""
    if force or not load_docs(seed.id):
        crawl_program(seed, fetcher)
    if force or seed.id not in Index.available():
        index_program(seed)
    extract_one(seed, force=force)


def _guard(fn):
    try:
        fn()
    except BudgetExceeded as e:
        print(f"\nSTOPPED (no cost incurred): {e}")
        raise typer.Exit(2)
    except MissingKeyError as e:
        print(f"\n{e}")
        raise typer.Exit(1)


@app.command("check-seeds")
def check_seeds():
    """Check that every seed/extra URL in programs.yaml is reachable."""
    fetcher = Fetcher(settings)
    for seed in load_programs():
        for url in seed.urls + seed.extra_urls:
            print(f"{fetcher.status(url):>20}  {seed.id:<18} {url}")


@app.command("crawl")
def crawl_cmd(program: Optional[str] = typer.Option(None), all_: bool = typer.Option(False, "--all")):
    """Discover and download relevant pages (no LLM, free)."""
    fetcher = Fetcher(settings)
    for seed in _select(program, all_):
        print(f"   {crawl_program(seed, fetcher)} documents kept")


@app.command("index")
def index_cmd(program: Optional[str] = typer.Option(None), all_: bool = typer.Option(False, "--all")):
    """Chunk + embed crawled pages locally (no LLM, free)."""
    for seed in _select(program, all_):
        index_program(seed)


@app.command("extract")
def extract_cmd(program: Optional[str] = typer.Option(None), all_: bool = typer.Option(False, "--all"),
                force: bool = typer.Option(False, help="Re-extract even if cached (uses Gemini quota)")):
    """Extract structured facts with Gemini (free tier, rate limited)."""
    def go():
        for seed in _select(program, all_):
            extract_one(seed, force)
    _guard(go)


@app.command("run")
def run_cmd(program: Optional[str] = typer.Option(None), all_: bool = typer.Option(False, "--all"),
            force: bool = typer.Option(False, help="Redo every step even if cached")):
    """crawl → index → extract. Resumable: finished steps are skipped."""
    def go():
        fetcher = Fetcher(settings)
        for seed in _select(program, all_):
            run_program(seed, force, fetcher)
    _guard(go)


@app.command("usage")
def usage_cmd():
    """Show Gemini usage today (Pacific time) per model vs. the daily budgets."""
    tracker = UsageTracker(settings.daily_token_budget, settings.daily_request_budget)
    for model in dict.fromkeys([settings.extract_model, settings.chat_model, *tracker.used()]):
        u = tracker.used(model)
        print(f"{model:<24} {u['requests']:>4} / {settings.daily_request_budget} requests, "
              f"{u['tokens']:>9,} / {settings.daily_token_budget:,} tokens today")


@app.command("report")
def report_cmd():
    """List missing / unverified fields per program."""
    for seed in load_programs():
        record = load_record(seed.id)
        if not record:
            print(f"{seed.id:<18} not extracted yet")
            continue
        gaps = missing_fields(record)
        print(f"{seed.id:<18} {len(gaps)} gaps: {', '.join(gaps) if gaps else '—'}")


if __name__ == "__main__":
    app()
