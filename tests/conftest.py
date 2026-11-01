import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import config, extract, index, ratelimit  # noqa: E402


class HashEmbedder:
    """Deterministic bag-of-words embedder: no model download, similar texts share words."""

    dim = 256

    def embed(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1
        return out


class FakeLLM:
    """Returns canned responses in order and records the prompts it received."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, *, model, json_mode=False, max_tokens=None):
        self.calls.append(messages)
        return self.responses.pop(0)


@pytest.fixture
def tmp_data(tmp_path, monkeypatch):
    """Redirect every data directory to a temp folder."""
    monkeypatch.setattr(index, "INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(extract, "EXTRACTED_DIR", tmp_path / "extracted")
    monkeypatch.setattr(config, "USAGE_FILE", tmp_path / "usage.json")
    monkeypatch.setattr(ratelimit, "USAGE_FILE", tmp_path / "usage.json")
    return tmp_path
