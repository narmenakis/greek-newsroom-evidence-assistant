"""Load and validate the article CSV before it reaches the vector store."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
from langchain_core.documents import Document


REQUIRED_COLUMNS = {"url", "title", "full-text", "author", "website", "datetime", "section"}
_NOT_FOUND_TITLE = re.compile(r"(?:σελίδα\s+δεν\s+βρέθηκε|page\s+not\s+found|404\s+not\s+found)", re.I)


class CorpusError(ValueError):
    """Raised when the corpus cannot safely be indexed."""


def canonicalize_url(value: str) -> str:
    value = str(value).strip()
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise CorpusError(f"Invalid article URL: {value!r}")
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise CorpusError(f"Invalid article URL: {value!r}")
    netloc = hostname
    try:
        port = parsed.port
    except ValueError as exc:
        raise CorpusError(f"Invalid article URL: {value!r}") from exc
    if port and port not in {80, 443}:
        netloc = f"{hostname}:{port}"
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit((parsed.scheme.lower(), netloc, path, parsed.query, ""))


def _timestamp(value: object, *, row_index: int, url: str) -> float:
    if value is None or pd.isna(value) or not str(value).strip():
        return 0.0
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        raise CorpusError(f"Row {row_index} has an invalid publication date for {url}: {value!r}")
    return float(parsed.timestamp())


def load_corpus(path: str | Path = "sample_database.csv") -> list[Document]:
    """Return validated LangChain documents with stable source metadata."""

    corpus_path = Path(path)
    if not corpus_path.is_file():
        raise CorpusError(f"Corpus file does not exist: {corpus_path}")
    try:
        frame = pd.read_csv(corpus_path)
    except Exception as exc:
        raise CorpusError(f"Could not read corpus {corpus_path}: {exc}") from exc
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise CorpusError(f"Corpus is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise CorpusError(f"Corpus is empty: {corpus_path}")

    documents: list[Document] = []
    seen_urls: set[str] = set()
    for row_index, row in frame.iterrows():
        try:
            url = canonicalize_url(row["url"])
        except CorpusError as exc:
            raise CorpusError(f"Row {row_index}: {exc}") from exc
        if url in seen_urls:
            raise CorpusError(f"Duplicate canonical URL at row {row_index}: {url}")
        seen_urls.add(url)

        title = "" if pd.isna(row["title"]) else str(row["title"]).strip()
        body = "" if pd.isna(row["full-text"]) else str(row["full-text"]).strip()
        if not title:
            raise CorpusError(f"Row {row_index} has an empty title: {url}")
        if not body:
            raise CorpusError(f"Row {row_index} has an empty article body: {url}")
        if _NOT_FOUND_TITLE.search(title):
            raise CorpusError(f"Row {row_index} appears to be a not-found page: {url}")

        website = "" if pd.isna(row["website"]) else str(row["website"]).strip()
        if not website:
            raise CorpusError(f"Row {row_index} has no website: {url}")
        article_id = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        metadata = {
            "article_id": article_id,
            "url": url,
            "title": title,
            "author": "" if pd.isna(row["author"]) else str(row["author"]).strip(),
            "website": website,
            "datetime": _timestamp(row["datetime"], row_index=row_index, url=url),
            "section": "" if pd.isna(row["section"]) else str(row["section"]).strip(),
            "row_index": int(row_index),
            "source_type": "archive",
        }
        documents.append(Document(page_content=body, metadata=metadata))
    return documents
