"""Turn retrieved chunks into structured, cited facts — one LLM call per program.

The Gemini free tier allows ~20 requests/day per model, so all field groups are filled in a single
call: chunks are retrieved separately for each group (so every topic is covered) and sent together.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from .config import EXTRACTED_DIR, Settings
from .index import Embedder, Index
from .llm import LLMClient
from .schema import FIELD_GROUPS, Fact, ProgramRecord, ProgramSeed, quote_in_text

TOP_K_PER_GROUP = 3
MAX_CHUNKS = 14  # ~7K tokens of sources per call

SYSTEM = """You extract facts about ONE specific master's program from official web pages.
Rules:
- Use ONLY the numbered sources given. Never use outside knowledge, never guess.
- Only report facts about the program named by the user — ignore other programs of the same school (MBA, other MSc, bachelor...).
- For every field return {"value": ..., "quote": ..., "source": n}:
  * "quote" = a short passage (max ~25 words) copied EXACTLY, character for character, from source n that supports the value.
  * "source" = the number of the source the quote comes from.
- If a field is not stated in the sources: {"value": null, "quote": null, "source": null}.
- Write values in English. Numbers must be plain numbers (no currency symbols, commas or %).
- Answer with a single JSON object and nothing else."""

_TYPE_HINT = {"number": "number or null", "text": "short text or null",
              "list": "list of short strings or null", "boolean": "true/false or null"}


def build_prompt(seed: ProgramSeed, group_names: list[str], chunks: list[dict]) -> list[dict]:
    sources = "\n\n".join(f"[Source {i}] {c['url']}\n{c['text']}" for i, c in enumerate(chunks, 1))
    template = {g: {name: {"value": f"<{_TYPE_HINT[typ]}: {desc}>", "quote": "<exact quote>", "source": "<n>"}
                    for name, (typ, desc) in FIELD_GROUPS[g]["fields"].items()}
                for g in group_names}
    user = (f"Program: {seed.program} — {seed.school} ({seed.country})\n\n"
            f"SOURCES:\n{sources}\n\n"
            f"Fill this JSON object:\n{json.dumps(template, indent=1, ensure_ascii=False)}")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def coerce(value, typ: str):
    if value is None or value == "" or value == []:
        return None
    if typ == "number":
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return value
        m = re.search(r"-?\d[\d,.\s]*", str(value))
        if not m:
            return None
        digits = re.sub(r"[,\s]", "", m.group()).rstrip(".")
        try:
            return float(digits)
        except ValueError:
            return None
    if typ == "list":
        return [str(v) for v in value] if isinstance(value, list) else [str(value)]
    if typ == "boolean":
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"yes", "true", "required", "oui"}
    return str(value) if not isinstance(value, list) else "; ".join(map(str, value))


def parse_group(data: dict, group: dict, chunks: list[dict], docs: list[dict]) -> dict[str, Fact]:
    data = data if isinstance(data, dict) else {}
    facts: dict[str, Fact] = {}
    for name, (typ, _) in group["fields"].items():
        item = data.get(name) or {}
        if not isinstance(item, dict):
            item = {"value": item}
        value = coerce(item.get("value"), typ)
        if value is None:
            facts[name] = Fact()
            continue
        quote = item.get("quote") if isinstance(item.get("quote"), str) else None
        try:
            chunk = chunks[int(item.get("source")) - 1]
            candidates = [chunk] + [c for c in chunks if c is not chunk]
        except (TypeError, ValueError, IndexError):
            candidates = chunks
        # The quote must appear verbatim in the full source document, otherwise it's flagged
        source_url, verified = candidates[0]["url"] if candidates else None, False
        for c in candidates:
            doc_text = docs[c["doc_id"]]["text"] if c["doc_id"] < len(docs) else c["text"]
            if quote_in_text(quote, doc_text):
                source_url, verified = c["url"], True
                break
        facts[name] = Fact(value=value, quote=quote, source_url=source_url, verified=verified)
    return facts


def retrieve_for_groups(seed: ProgramSeed, group_names: list[str], index: Index, embedder: Embedder) -> list[dict]:
    """Top chunks for each group's topic, merged without duplicates."""
    chunks, seen = [], set()
    for g in group_names:
        group = FIELD_GROUPS[g]
        for c in index.search(f"{seed.program} {group['query']}", embedder, k=TOP_K_PER_GROUP,
                              program_id=seed.id, boost_types=group["doc_types"]):
            key = (c["url"], c["text"][:80])
            if key not in seen:
                seen.add(key)
                chunks.append(c)
    return chunks[:MAX_CHUNKS]


def extract_groups(seed: ProgramSeed, group_names: list[str], index: Index, embedder: Embedder,
                   llm: LLMClient, docs: list[dict], settings: Settings) -> dict[str, dict[str, Fact]]:
    """Fill several field groups with ONE LLM call."""
    chunks = retrieve_for_groups(seed, group_names, index, embedder)
    if not chunks:  # nothing crawled for this program: don't spend a request
        return {g: {name: Fact() for name in FIELD_GROUPS[g]["fields"]} for g in group_names}
    messages = build_prompt(seed, group_names, chunks)
    for attempt in range(2):
        raw = llm.chat(messages, model=settings.extract_model, json_mode=True)
        try:
            data = json.loads(raw)
            break
        except json.JSONDecodeError:
            if attempt == 1:
                raise
    return {g: parse_group(data.get(g, {}), FIELD_GROUPS[g], chunks, docs) for g in group_names}


def record_path(program_id: str):
    return EXTRACTED_DIR / f"{program_id}.json"


def load_record(program_id: str) -> ProgramRecord | None:
    path = record_path(program_id)
    return ProgramRecord.model_validate_json(path.read_text(encoding="utf-8")) if path.exists() else None


def save_record(record: ProgramRecord) -> None:
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    record_path(record.seed.id).write_text(record.model_dump_json(indent=1), encoding="utf-8")


def extract_program(seed: ProgramSeed, index: Index, embedder: Embedder, llm: LLMClient,
                    docs: list[dict], settings: Settings, force: bool = False) -> ProgramRecord:
    """Extract the field groups not done yet (all of them with force) in one call, then save."""
    record = (None if force else load_record(seed.id)) or ProgramRecord(seed=seed)
    record.seed = seed  # pick up edits made to programs.yaml
    todo = [g for g in FIELD_GROUPS if force or g not in record.groups]
    if not todo:
        return record
    print(f"  extracting {', '.join(todo)}…")
    record.groups.update(extract_groups(seed, todo, index, embedder, llm, docs, settings))
    record.extracted_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    save_record(record)
    return record


def missing_fields(record: ProgramRecord) -> list[str]:
    out = []
    for group_name, group in FIELD_GROUPS.items():
        for name in group["fields"]:
            f = record.fact(group_name, name)
            if f.value is None:
                out.append(f"{group_name}.{name}")
            elif not f.verified:
                out.append(f"{group_name}.{name} (unverified)")
    return out
