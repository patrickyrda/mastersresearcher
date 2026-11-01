"""Crawl a program's site from its seed URLs, keeping only pages relevant to our fields."""
from __future__ import annotations

import heapq
import json
import re
from urllib.parse import urlparse

from .config import MANUAL_DIR, RAW_DIR, Settings
from .fetch import Document, Fetcher, html_to_text, pdf_to_text
from .schema import ProgramSeed

# Link keywords (matched against URL + anchor text) and their weights
KEYWORDS = {
    "admission": 3, "requirement": 3, "eligib": 3, "apply": 2, "application": 2, "deadline": 3,
    "tuition": 3, "fees": 3, "funding": 1, "scholarship": 1,
    "employment": 4, "career": 3, "placement": 3, "outcome": 3, "salar": 3, "report": 2,
    "class-profile": 4, "class profile": 4, "profile": 1, "student": 1, "faq": 2,
    "programme": 1, "program": 1, "curriculum": 1,
    # French
    "candidature": 3, "frais": 3, "carriere": 3, "carrière": 3, "debouches": 3, "débouchés": 3,
    "insertion": 3, "admissibilit": 3,
}
NEGATIVE = ["mba", "minor", "emba", "executive", "phd", "doctor", "bachelor", "bba", "undergrad", "news",
            "event", "blog", "press", "podcast", "login", "faculty", "research", "linkedin",
            "facebook", "twitter", "instagram", "youtube", "x.com", "privacy", "cookie", "legal",
            "mentions-legales", "sitemap", "/search", "calendar", "webinar", "summer",
            "short-course", "online-course", "certificate", "exchange"]

DOC_TYPES = {
    "outcomes": ["employment", "career", "placement", "outcome", "salar", "debouches", "débouchés",
                 "insertion", "carriere", "carrière", "job"],
    "class_profile": ["class-profile", "class profile", "class_profile", "profile", "cohort", "students"],
    "admissions": ["admission", "apply", "application", "requirement", "eligib", "deadline", "candidature"],
    "fees": ["tuition", "fees", "frais", "funding", "scholarship", "cost"],
}

MIN_LINK_SCORE = 2
MIN_OFF_SECTION_SCORE = 5  # pages outside the program's section must look clearly relevant
MAX_OFF_SECTION_PAGES = 5


def site_key(netloc: str) -> str:
    """Registered domain, e.g. www.jbs.cam.ac.uk -> cam.ac.uk, mitsloan.mit.edu -> mit.edu."""
    labels = netloc.lower().split(":")[0].split(".")
    if len(labels) >= 3 and labels[-2] in {"ac", "co", "org", "gov"} and len(labels[-1]) == 2:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])     


def score_link(url: str, anchor: str, seed_paths: list[str]) -> int:
    haystack = (url + " " + anchor).lower()
    if any(neg in haystack for neg in NEGATIVE):
        return -1
    score = sum(w for kw, w in KEYWORDS.items() if kw in haystack)
    path = urlparse(url).path.rstrip("/")
    for seed_path in filter(None, seed_paths):
        if path == seed_path or path.startswith(seed_path + "/"):
            return score + 5  # inside the program's own section of the site
        parent = seed_path.rsplit("/", 1)[0]
        if parent and path.startswith(parent + "/"):
            return -1  # a sibling program (e.g. /masters/mim next to /masters/mif): wrong data
    return score


_DEGREE_PREFIX = re.compile(r"^(?:(?:the|master|masters|master-s|msc|ms|mfin|mphil|m1|m2)(?:-in|-of|-en)?-)+")


def program_slug(seed_path: str) -> str:
    """Identifying part of a program URL: /en/master-s-programs/master-international-finance -> international-finance."""
    slug = _DEGREE_PREFIX.sub("", seed_path.rstrip("/").rsplit("/", 1)[-1].lower())
    return slug if len(slug) >= 6 and "-" in slug else ""


def in_section(url: str, seed_paths: list[str]) -> bool:
    """Is this page part of the program itself (under its path, or naming it in the URL)?"""
    path = urlparse(url).path.rstrip("/").lower()
    for p in filter(None, seed_paths):
        slug = program_slug(p)
        if path == p.lower() or path.startswith(p.lower() + "/") or (slug and f"-{slug}" in f"-{path}"):
            return True
    return False


def is_school_wide(url: str) -> bool:
    """Shallow pages like /en/admissions/deadlines; deep paths belong to some other program."""
    return len([p for p in urlparse(url).path.split("/") if p]) <= 3


def _seed_path(url: str) -> str:
    """Path of a seed URL without trailing slash or file name (index.php, default.aspx...)."""
    parts = [p for p in urlparse(url).path.split("/") if p]
    if parts and "." in parts[-1]:
        parts = parts[:-1]
    return "/" + "/".join(parts) if parts else ""


def classify(url: str, title: str = "") -> str:
    haystack = (url + " " + title).lower()
    for doc_type, words in DOC_TYPES.items():
        if any(w in haystack for w in words):
            return doc_type
    return "general"


def manual_documents(program_id: str, root=MANUAL_DIR) -> list[Document]:
    """Files the user saved from their browser (Ctrl+S / print to PDF) for sites that block crawlers."""
    docs = []
    folder = root / program_id
    for path in sorted(folder.glob("*")) if folder.exists() else []:
        suffix = path.suffix.lower()
        url = f"manual/{program_id}/{path.name}"
        if suffix == ".pdf":
            docs.append(Document(url, "pdf", pdf_to_text(path.read_bytes())))
        elif suffix in {".html", ".htm"}:
            text, title, _ = html_to_text(path.read_text(encoding="utf-8", errors="replace"), "https://local/")
            docs.append(Document(url, "html", text, title=title))
        elif suffix in {".txt", ".md"}:
            docs.append(Document(url, "html", path.read_text(encoding="utf-8", errors="replace")))
    return docs


def crawl(seed: ProgramSeed, fetcher: Fetcher, settings: Settings) -> list[dict]:
    sites = {site_key(urlparse(u).netloc) for u in seed.urls + seed.extra_urls}
    seed_paths = [_seed_path(u) for u in seed.urls]
    docs: list[dict] = []
    seen: set[str] = set()

    texts: set[int] = set()

    def keep(doc: Document, forced_type: str | None = None) -> bool:
        """Store a document; False if it's too short or a duplicate of one we already have."""
        key = hash(" ".join(doc.text.split()))
        if len(doc.text) < 200 or key in texts:
            return False
        texts.add(key)
        docs.append({"url": doc.url, "title": doc.title, "content_type": doc.content_type,
                     "doc_type": forced_type or classify(doc.url, doc.title),
                     "fetched_at": doc.fetched_at, "text": doc.text})
        return True

    # 1. Hand-picked URLs are always fetched and don't count towards the page limit
    for url in seed.extra_urls:
        seen.add(url)
        if doc := fetcher.get(url):
            keep(doc)

    # 2. Pages saved by hand (sites that block bots): manual/<program-id>/*.html|pdf|txt
    for doc in manual_documents(seed.id):
        keep(doc)
        print(f"  [{len(docs):>2}] {doc.url}")

    # 3. Best-first crawl from the seed pages. Pages outside the program's own section of the
    #    site (school-wide admissions/careers pages) need a higher score, are capped, and are
    #    never expanded further — that's where crawls drift into other programs.
    frontier: list[tuple[int, int, str]] = [(-100, 0, u) for u in seed.urls]  # (-score, depth, url)
    heapq.heapify(frontier)
    fetched = off_section = 0
    while frontier and fetched < settings.crawl_max_pages:
        _, depth, url = heapq.heappop(frontier)
        if url in seen:
            continue
        inside = depth == 0 or in_section(url, seed_paths)
        if not inside and off_section >= MAX_OFF_SECTION_PAGES:
            continue
        seen.add(url)
        doc = fetcher.get(url)
        fetched += 1
        if not doc:
            continue
        if depth == 0 and doc.final_url and doc.final_url != url:
            seed_paths.append(_seed_path(doc.final_url))  # seed redirected: its new path is the section too
        if not keep(doc, "general" if depth == 0 else None):
            fetched -= 1  # duplicates (?entry=true, tracking variants...) don't use the page budget
            continue
        off_section += not inside
        print(f"  [{len(docs):>2}] {url}")
        if doc.content_type != "html" or depth >= settings.crawl_depth or not inside:
            continue
        for link, anchor in doc.links:
            if link in seen or site_key(urlparse(link).netloc) not in sites:
                continue
            score = score_link(link, anchor, seed_paths)
            if in_section(link, seed_paths):
                ok = score >= MIN_LINK_SCORE
            else:
                ok = score >= MIN_OFF_SECTION_SCORE and is_school_wide(link)
            if ok:
                heapq.heappush(frontier, (-score, depth + 1, link))
    return docs


def docs_path(program_id: str):
    return RAW_DIR / program_id / "docs.json"


def save_docs(program_id: str, docs: list[dict]) -> None:
    path = docs_path(program_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(docs, ensure_ascii=False, indent=1), encoding="utf-8")


def load_docs(program_id: str) -> list[dict]:
    path = docs_path(program_id)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
