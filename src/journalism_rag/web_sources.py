"""Normalized, bounded web evidence for live research workflows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


MAX_WEB_EXCERPT_CHARACTERS = 20_000
_TRACKING_QUERY_PREFIXES = ("utm_",)
_TRACKING_QUERY_KEYS = frozenset({"fbclid", "gclid", "dclid", "msclkid"})


def normalize_web_url(url: str) -> str:
    """Return a stable URL suitable for comparison and citation display.

    Only HTTP(S) URLs with a hostname are accepted. Fragments and common
    analytics parameters do not identify a different article, while all other
    query parameters are retained and sorted deterministically.
    """

    if not isinstance(url, str) or not url.strip():
        raise ValueError("web URL cannot be empty")
    parsed = urlsplit(url.strip())
    scheme = parsed.scheme.casefold()
    if scheme not in {"http", "https"}:
        raise ValueError("web URL must use http or https")
    if not parsed.hostname:
        raise ValueError("web URL must include a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("web URL cannot include credentials")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("web URL has an invalid port") from exc

    hostname = parsed.hostname.casefold().rstrip(".")
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname
    if port is not None and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        netloc = f"{netloc}:{port}"

    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/") or "/"
    query_pairs = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.casefold() not in _TRACKING_QUERY_KEYS
        and not key.casefold().startswith(_TRACKING_QUERY_PREFIXES)
    ]
    query_pairs.sort()
    return urlunsplit((scheme, netloc, path, urlencode(query_pairs), ""))


def _utc_datetime(value: datetime, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class WebEvidence:
    """One bounded, normalized result obtained from a web source."""

    title: str
    url: str
    excerpt: str
    retrieved_at: datetime
    published_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.title, str) or not self.title.strip():
            raise ValueError("web evidence title cannot be empty")
        excerpt = self.excerpt.strip() if isinstance(self.excerpt, str) else ""
        if not excerpt:
            raise ValueError("web evidence excerpt cannot be empty")
        if len(excerpt) > MAX_WEB_EXCERPT_CHARACTERS:
            raise ValueError(
                "web evidence excerpt exceeds "
                f"{MAX_WEB_EXCERPT_CHARACTERS} characters"
            )
        object.__setattr__(self, "title", self.title.strip())
        object.__setattr__(self, "url", normalize_web_url(self.url))
        object.__setattr__(self, "excerpt", excerpt)
        object.__setattr__(
            self, "retrieved_at", _utc_datetime(self.retrieved_at, "retrieved_at")
        )
        if self.published_at is not None:
            object.__setattr__(
                self, "published_at", _utc_datetime(self.published_at, "published_at")
            )


def deduplicate_web_evidence(
    evidence: tuple[WebEvidence, ...] | list[WebEvidence],
) -> tuple[WebEvidence, ...]:
    """Keep the first result for each normalized URL in input order."""

    unique: list[WebEvidence] = []
    seen_urls: set[str] = set()
    for item in evidence:
        if item.url in seen_urls:
            continue
        seen_urls.add(item.url)
        unique.append(item)
    return tuple(unique)
