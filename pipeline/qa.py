"""Free-form questions over the collected pages (RAG) with source citations."""
from __future__ import annotations

from .config import Settings
from .index import Embedder, Index
from .llm import LLMClient
from .schema import ProgramSeed

TOP_K = 8

SYSTEM = """You help a student compare finance master's programs.
Answer ONLY from the numbered sources. Cite sources inline like [2]. If the sources don't contain the
answer, say so plainly — never guess numbers. Be concise; use a short list or table when comparing programs."""


def answer(question: str, index: Index, embedder: Embedder, llm: LLMClient, settings: Settings,
           seeds: dict[str, ProgramSeed], program_id: str | None = None) -> tuple[str, list[dict]]:
    chunks = index.search(question, embedder, k=TOP_K, program_id=program_id)
    if not chunks:
        return "No indexed pages yet — run the pipeline first.", []

    def label(c: dict) -> str:
        s = seeds.get(c["program_id"])
        return f"{s.program} — {s.school}" if s else c["program_id"]

    sources = "\n\n".join(f"[{i}] ({label(c)}) {c['url']}\n{c['text']}" for i, c in enumerate(chunks, 1))
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"SOURCES:\n{sources}\n\nQUESTION: {question}"}]
    return llm.chat(messages, model=settings.chat_model, max_tokens=1500), chunks
