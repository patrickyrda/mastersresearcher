"""Data models: program seeds, extracted facts, and the field groups we extract."""
from __future__ import annotations

import re
import unicodedata
from typing import Literal, Union

import yaml
from pydantic import BaseModel, Field

from .config import PROGRAMS_FILE

Value = Union[str, float, int, bool, list[str], None]


class ProgramSeed(BaseModel):
    id: str
    school: str
    program: str
    country: Literal["US", "FR", "UK"]
    city: str = ""
    ranking: str | None = None  # e.g. "FT MiF 2025 #3" — fill in by hand from the published ranking
    urls: list[str]  # program home page(s): crawl starts here
    extra_urls: list[str] = []  # pages/PDFs always fetched (employment report, class profile...)


class Fact(BaseModel):
    value: Value = None
    quote: str | None = None
    source_url: str | None = None
    verified: bool = False  # quote found verbatim in the source text


class ProgramRecord(BaseModel):
    seed: ProgramSeed
    extracted_at: str | None = None
    groups: dict[str, dict[str, Fact]] = Field(default_factory=dict)

    def fact(self, group: str, name: str) -> Fact:
        return self.groups.get(group, {}).get(name, Fact())


def load_programs(path=PROGRAMS_FILE) -> list[ProgramSeed]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [ProgramSeed(**p) for p in data["programs"]]


# Each group = one LLM call per program. `query` drives retrieval, `doc_types` boosts matching pages.
# Field spec: name -> (type, description). Types: number | text | list | boolean
FIELD_GROUPS: dict[str, dict] = {
    "overview": {
        "query": "program overview duration length language of instruction tuition fees cost campus",
        "doc_types": ["general", "fees"],
        "fields": {
            "duration": ("text", "Program length, e.g. '12 months', '2 years'"),
            "language": ("text", "Language(s) of instruction"),
            "tuition": ("number", "Total tuition fee for the whole program (number only)"),
            "tuition_currency": ("text", "Currency code of tuition: EUR, GBP or USD"),
            "start_date": ("text", "When the program starts, e.g. 'September'"),
            "format": ("text", "Full-time / part-time, campus"),
        },
    },
    "admissions": {
        "query": "admission requirements eligibility prerequisites GMAT GRE TOEFL IELTS deadlines application rounds fee",
        "doc_types": ["admissions"],
        "fields": {
            "degree_required": ("text", "Required prior degree / academic background"),
            "quant_requirements": ("text", "Required quantitative / maths / finance background"),
            "work_experience": ("text", "Work experience required or expected"),
            "gmat_gre_policy": ("text", "Is GMAT/GRE required, optional or waived? Include any conditions"),
            "gmat_gre_minimum": ("text", "Minimum or recommended GMAT/GRE score if stated"),
            "english_test_minimum": ("text", "Minimum TOEFL / IELTS / other English test scores"),
            "deadlines": ("list", "Application rounds and their deadlines, one item per round"),
            "application_fee": ("text", "Application fee with currency"),
        },
    },
    "documents": {
        "query": "application documents required CV resume transcripts recommendation letters essays video interview",
        "doc_types": ["admissions"],
        "fields": {
            "cv": ("boolean", "Is a CV / resume required?"),
            "transcripts": ("boolean", "Are academic transcripts required?"),
            "recommendation_letters": ("text", "Number/type of recommendation letters"),
            "essays": ("list", "Essays / motivation letter / questions required"),
            "video_or_interview": ("text", "Video essay and/or interview requirements"),
            "other_documents": ("list", "Any other required documents (ID, test reports, ...)"),
        },
    },
    "outcomes": {
        "query": "employment report graduates employed within 3 months average salary median base salary sectors employers careers placement",
        "doc_types": ["outcomes"],
        "fields": {
            "employment_rate": ("number", "Percentage of graduates employed (number only, e.g. 95)"),
            "employment_rate_window": ("text", "When measured, e.g. 'within 3 months of graduation'"),
            "avg_salary": ("number", "Average or median BASE salary of graduates (number only)"),
            "salary_currency": ("text", "Currency code of salary: EUR, GBP or USD"),
            "salary_definition": ("text", "Average vs median, base vs total, which year's class"),
            "report_year": ("text", "Year / class of the employment report"),
            "top_sectors": ("list", "Main sectors or job functions graduates go into"),
            "top_employers": ("list", "Named employers that hired graduates"),
        },
    },
    "class_profile": {
        "query": "class profile students cohort size average age nationalities international GMAT average academic background work experience",
        "doc_types": ["class_profile", "outcomes"],
        "fields": {
            "class_size": ("number", "Number of students in the cohort"),
            "average_age": ("number", "Average age of students"),
            "pct_international": ("number", "Percentage of international students (number only)"),
            "average_gmat_gre": ("text", "Average GMAT/GRE of admitted students"),
            "academic_backgrounds": ("list", "Undergraduate backgrounds / majors of students"),
            "average_work_experience": ("text", "Average prior work experience"),
        },
    },
}

_QUOTE_CHARS = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                              "–": "-", "—": "-", " ": " ", " ": " "})


def normalize(text: str) -> str:
    """Normalize text for quote matching: unicode, quotes/dashes, case, whitespace."""
    text = unicodedata.normalize("NFKC", text).translate(_QUOTE_CHARS).lower()
    text = re.sub(r"[*_#|`]", " ", text)  # markdown/table noise from extraction
    return re.sub(r"\s+", " ", text).strip()


def quote_in_text(quote: str | None, text: str) -> bool:
    if not quote or len(quote.strip()) < 3:
        return False
    return normalize(quote) in normalize(text)
