import json
import re
from pathlib import Path

import pytest

from conftest import FakeLLM, HashEmbedder
from pipeline.config import Settings
from pipeline.discover import MIN_LINK_SCORE, _seed_path, classify, in_section, is_school_wide, program_slug, manual_documents, score_link, site_key
from pipeline.extract import coerce, extract_program, load_record, missing_fields
from pipeline.index import Index, build_index, chunk_text
from pipeline.schema import FIELD_GROUPS, ProgramSeed, load_programs, quote_in_text

ROOT = Path(__file__).resolve().parent.parent

SEED = ProgramSeed(id="test-mif", school="Test School", program="Master in Finance", country="FR",
                   urls=["https://www.test.edu/master-finance"])
DOCS = [
    {"url": "https://www.test.edu/master-finance/admissions", "doc_type": "admissions",
     "text": "Admission requirements for the Master in Finance. The GMAT or GRE is required for all "
             "applicants. Minimum TOEFL iBT score: 100. Two recommendation letters are needed. " * 3},
    {"url": "https://www.test.edu/master-finance/careers", "doc_type": "outcomes",
     "text": "Employment report 2025: 96% of graduates were employed within 3 months. "
             "The average base salary was €78,500. Top employers: J.P. Morgan, BNP Paribas. " * 3},
]


# ---------------- schema / text helpers ----------------

def test_quote_matching_is_whitespace_case_and_quote_insensitive():
    text = "The  average base salary\nwas €78,500 — up from last year’s class."
    assert quote_in_text("average base salary was €78,500", text)
    assert quote_in_text("up from last year's class", text)
    assert not quote_in_text("average salary was €90,000", text)
    assert not quote_in_text("", text)


def test_coerce_types():
    assert coerce("€78,500", "number") == 78500.0
    assert coerce("96%", "number") == 96.0
    assert coerce("n/a", "number") is None
    assert coerce("CV", "list") == ["CV"]
    assert coerce("Yes", "boolean") is True
    assert coerce(None, "text") is None


def test_chunk_text_overlaps_and_covers_everything():
    words = [f"w{i}" for i in range(1000)]
    chunks = chunk_text(" ".join(words), size=300, overlap=50)
    assert chunks[0].split()[0] == "w0"
    assert chunks[-1].split()[-1] == "w999"
    assert chunks[1].split()[0] == "w250"


def test_programs_yaml_is_valid():
    seeds = load_programs(ROOT / "programs.yaml")
    assert len(seeds) >= 20
    assert len({s.id for s in seeds}) == len(seeds)
    assert {s.country for s in seeds} == {"US", "FR", "UK"}


# ---------------- crawling heuristics ----------------

def test_site_key():
    assert site_key("www.jbs.cam.ac.uk") == "cam.ac.uk"
    assert site_key("mitsloan.mit.edu") == "mit.edu"
    assert site_key("master.edhec.edu") == "edhec.edu"


def test_link_scoring_prefers_relevant_pages():
    seed_paths = ["/master-finance"]
    good = score_link("https://x.edu/master-finance/admissions-requirements", "Admissions", seed_paths)
    career = score_link("https://x.edu/careers/employment-report", "Employment report", seed_paths)
    mba = score_link("https://x.edu/mba/admissions", "MBA admissions", seed_paths)
    assert good >= MIN_LINK_SCORE and career >= MIN_LINK_SCORE
    assert mba < 0
    sibling = score_link("https://x.edu/masters/mim/admissions", "Admissions", ["/masters/mif"])
    own = score_link("https://x.edu/masters/mif/admissions", "Admissions", ["/masters/mif"])
    school_wide = score_link("https://x.edu/admissions/deadlines", "Deadlines", ["/masters/mif"])
    assert sibling < 0 < school_wide < own
    assert _seed_path("https://x.edu/programs/ms-in-finance/index.php") == "/programs/ms-in-finance"
    assert classify("https://x.edu/careers/employment-report") == "outcomes"
    assert classify("https://x.edu/admissions/apply") == "admissions"


# ---------------- extraction (mocked LLM, no network) ----------------

def _answer(facts_by_group: dict) -> str:
    empty = {"value": None, "quote": None, "source": None}
    return json.dumps({g: {name: facts_by_group.get(g, {}).get(name, empty) for name in spec["fields"]}
                       for g, spec in FIELD_GROUPS.items()})


def test_extraction_verifies_quotes_and_resumes(tmp_data):
    emb = HashEmbedder()
    build_index(SEED.id, DOCS, emb)
    idx = Index.load([SEED.id])

    facts = {
        "admissions": {
            "gmat_gre_policy": {"value": "Required", "quote": "The GMAT or GRE is required", "source": 1},
            # hallucinated: quote doesn't exist in any source
            "application_fee": {"value": "€150", "quote": "application fee is €150", "source": 1},
        },
        # "source": 1 may point at the wrong chunk (numbering follows retrieval rank) — the verifier
        # must still find the quote in the other retrieved sources and attribute the right URL
        "outcomes": {"avg_salary": {"value": 78500, "quote": "average base salary was €78,500", "source": 1}},
    }
    llm = FakeLLM([_answer(facts)])
    record = extract_program(SEED, idx, emb, llm, DOCS, Settings())
    assert len(llm.calls) == 1  # all field groups in ONE request (free tier: ~20 requests/day per model)
    assert set(record.groups) == set(FIELD_GROUPS)

    gmat = record.fact("admissions", "gmat_gre_policy")
    assert gmat.value == "Required" and gmat.verified
    assert gmat.source_url.endswith("/admissions")

    salary = record.fact("outcomes", "avg_salary")
    assert salary.value == 78500 and salary.verified
    assert salary.source_url.endswith("/careers")

    fee = record.fact("admissions", "application_fee")
    assert fee.value == "€150" and not fee.verified  # flagged, not trusted
    assert "admissions.application_fee (unverified)" in missing_fields(record)

    # prompt contains only this program's sources and asks for exact quotes
    prompt = llm.calls[0][1]["content"]
    assert "Master in Finance — Test School" in prompt and "[Source 1]" in prompt

    # cached: a second run makes no LLM call
    again = FakeLLM([])
    extract_program(SEED, idx, emb, again, DOCS, Settings())
    assert again.calls == []
    assert load_record(SEED.id).fact("outcomes", "avg_salary").value == 78500


def test_retrieval_filters_by_program_and_boosts_doc_type(tmp_data):
    emb = HashEmbedder()
    build_index(SEED.id, DOCS, emb)
    other = [{"url": "https://other.edu/x", "doc_type": "outcomes", "text": "average base salary " * 50}]
    build_index("other", other, emb)
    idx = Index.load([SEED.id, "other"])
    hits = idx.search("average base salary employed", emb, k=3, program_id=SEED.id, boost_types=["outcomes"])
    assert hits and all(h["program_id"] == SEED.id for h in hits)
    assert hits[0]["doc_type"] == "outcomes"


# ---------------- zero-cost guarantee ----------------

# The only LLM provider allowed is the Gemini API (free tier). Any other hosted-LLM SDK could bill.
OTHER_LLM_SDKS = ["anthropic", "openai", "groq", "cohere", "mistralai", "langchain", "google.generativeai"]


def test_only_gemini_is_used_as_llm_provider():
    reqs = [line.split(">")[0].split("=")[0].strip().lower()
            for line in (ROOT / "requirements.txt").read_text().splitlines() if line.strip()]
    assert "google-genai" in reqs
    for sdk in OTHER_LLM_SDKS:
        assert sdk not in reqs
    for py in list((ROOT / "pipeline").glob("*.py")) + list((ROOT / "app").glob("*.py")):
        src = py.read_text(encoding="utf-8")
        for sdk in OTHER_LLM_SDKS:
            assert not re.search(rf"^\s*(import|from)\s+{re.escape(sdk)}\b", src, re.M), f"{sdk} in {py.name}"


def test_env_is_gitignored_and_key_not_in_repr():
    assert ".env" in (ROOT / ".gitignore").read_text().splitlines()
    s = Settings(gemini_api_key="AIzaSecret123")
    assert "AIzaSecret123" not in repr(s)
