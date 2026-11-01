"""Chunk documents, embed them locally (fastembed), and do cosine retrieval.

The index is small (a few thousand chunks), so it's plain numpy + JSON per program —
no vector database needed.
"""
from __future__ import annotations

import json
from typing import Protocol

import numpy as np

from .config import INDEX_DIR, Settings

CHUNK_WORDS = 350  # ≈ 500 tokens
OVERLAP_WORDS = 60


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class FastEmbedder:
    """Local CPU embeddings. Downloads the model (~200 MB) once on first use."""

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array(list(self.model.embed(texts)), dtype=np.float32)


_embedder: Embedder | None = None


def get_embedder(settings: Settings) -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = FastEmbedder(settings.embed_model)
    return _embedder


def chunk_text(text: str, size: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words) - overlap, 1), step)]


def _normalize_rows(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


def build_index(program_id: str, docs: list[dict], embedder: Embedder) -> int:
    chunks = []
    for doc_id, doc in enumerate(docs):
        for text in chunk_text(doc["text"]):
            chunks.append({"program_id": program_id, "doc_id": doc_id, "url": doc["url"],
                           "doc_type": doc["doc_type"], "text": text})
    out = INDEX_DIR / program_id
    out.mkdir(parents=True, exist_ok=True)
    vectors = embedder.embed([c["text"] for c in chunks]) if chunks else np.zeros((0, 1), np.float32)
    np.save(out / "vectors.npy", _normalize_rows(vectors))
    (out / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    return len(chunks)


class Index:
    """Chunks + vectors for one or more programs, loaded from disk."""

    def __init__(self, chunks: list[dict], vectors: np.ndarray):
        self.chunks = chunks
        self.vectors = vectors

    @classmethod
    def load(cls, program_ids: list[str]) -> "Index":
        chunks, vecs = [], []
        for pid in program_ids:
            d = INDEX_DIR / pid
            if not (d / "chunks.json").exists():
                continue
            c = json.loads((d / "chunks.json").read_text(encoding="utf-8"))
            if c:
                chunks += c
                vecs.append(np.load(d / "vectors.npy"))
        return cls(chunks, np.vstack(vecs) if vecs else np.zeros((0, 1), np.float32))

    @staticmethod
    def available() -> list[str]:
        return sorted(p.name for p in INDEX_DIR.glob("*") if (p / "chunks.json").exists()) if INDEX_DIR.exists() else []

    def search(self, query: str, embedder: Embedder, k: int = 6, program_id: str | None = None,
               boost_types: list[str] | None = None) -> list[dict]:
        if not self.chunks:
            return []
        q = _normalize_rows(embedder.embed([query]))[0]
        scores = self.vectors @ q
        if boost_types:
            scores = scores + np.array([0.08 if c["doc_type"] in boost_types else 0 for c in self.chunks])
        if program_id:
            scores = np.where([c["program_id"] == program_id for c in self.chunks], scores, -np.inf)
        order = np.argsort(-scores)[:k]
        return [self.chunks[i] | {"score": float(scores[i])} for i in order if np.isfinite(scores[i])]
