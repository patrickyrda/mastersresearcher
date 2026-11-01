"""Polite HTTP fetching with a disk cache, robots.txt checks, and HTML/PDF → text."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urldefrag, urlencode, urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import httpx

from .config import RAW_DIR, Settings

CACHE_DIR = RAW_DIR / "_cache"


@dataclass
class Document:
    url: str
    content_type: str  # "html" | "pdf"
    text: str
    links: list[tuple[str, str]] = field(default_factory=list)  # (absolute url, anchor text)
    title: str = ""
    fetched_at: str = ""
    final_url: str = ""  # after redirects


def clean_url(url: str) -> str:
    """Drop #fragments and tracking parameters so the same page isn't crawled twice."""
    parts = urlparse(urldefrag(url)[0])
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query) if not k.lower().startswith(("utm_", "gclid", "fbclid"))])
    return urlunparse(parts._replace(query=query))


def html_to_text(html: str, url: str) -> tuple[str, str, list[tuple[str, str]]]:
    """Return (main text, title, links) for an HTML page."""
    import lxml.html
    import trafilatura

    text = trafilatura.extract(html, url=url, include_tables=True, include_links=False,
                               favor_recall=True) or ""
    title, links = "", []
    try:
        tree = lxml.html.fromstring(html)
        title = (tree.findtext(".//title") or "").strip()
        for a in tree.xpath("//a[@href]"):
            href = a.get("href", "").strip()
            if not href or href.startswith(("mailto:", "tel:", "javascript:", "#")):
                continue
            absolute = clean_url(urljoin(url, href))
            if absolute.startswith("http"):
                links.append((absolute, " ".join(a.text_content().split())[:120]))
        if not text:
            text = " ".join(tree.text_content().split())
    except Exception:
        pass
    return text, title, links


def pdf_to_text(data: bytes) -> str:
    import pymupdf

    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        return "\n".join(page.get_text() for page in pdf)


class Fetcher:
    def __init__(self, settings: Settings, cache_dir: Path = CACHE_DIR, client: httpx.Client | None = None):
        self.settings = settings
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.client = client or httpx.Client(
            headers={"User-Agent": settings.user_agent, "Accept-Language": "en,fr;q=0.8"},
            follow_redirects=True, timeout=30,
        )
        self._robots: dict[str, RobotFileParser | None] = {}
        self._last_hit: dict[str, float] = {}

    def _key(self, url: str) -> str:
        return hashlib.sha1(url.encode()).hexdigest()

    def allowed(self, url: str) -> bool:
        parts = urlparse(url)
        host = f"{parts.scheme}://{parts.netloc}"
        if host not in self._robots:
            parser = None
            try:
                r = self.client.get(host + "/robots.txt")
                if r.status_code == 200:
                    parser = RobotFileParser()
                    parser.parse(r.text.splitlines())
            except httpx.HTTPError:
                pass
            self._robots[host] = parser
        parser = self._robots[host]
        return parser is None or parser.can_fetch(self.settings.user_agent, url)

    def _polite_wait(self, url: str) -> None:
        host = urlparse(url).netloc
        wait = self.settings.crawl_delay - (time.monotonic() - self._last_hit.get(host, 0))
        if wait > 0:
            time.sleep(wait)
        self._last_hit[host] = time.monotonic()

    def _download(self, url: str) -> tuple[bytes, str, str] | None:
        meta_path = self.cache_dir / f"{self._key(url)}.json"
        body_path = self.cache_dir / f"{self._key(url)}.bin"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta["status"] != 200:
                return None
            return body_path.read_bytes(), meta["content_type"], meta.get("final_url", url)

        if not self.allowed(url):
            meta = {"url": url, "status": "robots-disallowed"}
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            return None
        self._polite_wait(url)
        try:
            r = self.client.get(url)
        except httpx.HTTPError as e:
            print(f"  ! {url}: {type(e).__name__}")
            return None  # not cached: transient errors are retried next run
        ctype = r.headers.get("content-type", "").split(";")[0].strip().lower()
        meta = {"url": url, "final_url": str(r.url), "status": r.status_code, "content_type": ctype,
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        if r.status_code != 200:
            print(f"  ! {url}: HTTP {r.status_code}")
            return None
        body_path.write_bytes(r.content)
        return r.content, ctype, str(r.url)

    def get(self, url: str) -> Document | None:
        got = self._download(url)
        if not got:
            return None
        body, ctype, final_url = got
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if "pdf" in ctype or url.lower().endswith(".pdf"):
            try:
                return Document(url, "pdf", pdf_to_text(body), fetched_at=fetched_at, final_url=final_url)
            except Exception as e:
                print(f"  ! {url}: unreadable PDF ({type(e).__name__})")
                return None
        if "html" in ctype or not ctype:
            text, title, links = html_to_text(body.decode("utf-8", errors="replace"), final_url)
            return Document(url, "html", text, links, title, fetched_at, final_url)
        return None

    def status(self, url: str) -> str:
        """Fetch (or read cached) and report a short status string — used by `check-seeds`."""
        if self.get(url):
            return "ok"
        meta_path = self.cache_dir / f"{self._key(url)}.json"
        if meta_path.exists():
            return str(json.loads(meta_path.read_text(encoding="utf-8")).get("status"))
        return "network-error"
