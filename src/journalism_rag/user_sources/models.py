"""Typed, bounded lifecycle for session-local uploaded sources."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Callable


class SourceType(str, Enum):
    """The provenance category for an uploaded source."""

    USER_UPLOAD = "user_upload"


class SourceNotFoundError(LookupError):
    """Raised when a requested uploaded source is not in the session."""


@dataclass(frozen=True, slots=True)
class UploadLimits:
    """Safety limits applied before a source enters the session registry."""

    max_sources: int = 20
    # Five megabytes (decimal) per uploaded file for the current local phase.
    max_source_bytes: int = 5_000_000
    max_extracted_characters: int = 500_000
    max_chunks_per_source: int = 2_048
    supported_media_types: frozenset[str] = frozenset(
        {"text/plain", "text/markdown", "application/pdf"}
    )

    def validate(self) -> None:
        for name in (
            "max_sources",
            "max_source_bytes",
            "max_extracted_characters",
            "max_chunks_per_source",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if not self.supported_media_types:
            raise ValueError("supported_media_types cannot be empty")


@dataclass(frozen=True, slots=True)
class ExtractionMetadata:
    """Metadata describing the bounded text extraction result."""

    media_type: str
    original_filename: str | None
    source_bytes: int
    extracted_characters: int


@dataclass(frozen=True, slots=True)
class UploadedSource:
    """One source available to the current local application session."""

    source_id: str
    display_name: str
    content: str
    content_hash: str
    source_type: SourceType
    created_at: datetime
    extraction: ExtractionMetadata


@dataclass(frozen=True, slots=True)
class SourceAddResult:
    """Result of adding a source, including whether it was a duplicate."""

    source: UploadedSource
    created: bool


def _required_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} cannot be empty")
    return value.strip()


class InMemorySourceStore:
    """Session-local source registry with deterministic duplicate handling.

    This store deliberately has no owner or authentication concept.  Its
    lifetime is the current application session, and its records are separate
    from the permanent corpus.  A later index adapter can consume these
    records without changing their IDs or provenance metadata.
    """

    def __init__(
        self,
        *,
        limits: UploadLimits | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.limits = limits or UploadLimits()
        self.limits.validate()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._sources: dict[str, UploadedSource] = {}
        self._hash_to_id: dict[str, str] = {}

    def add_source(
        self,
        display_name: str,
        content: str,
        *,
        media_type: str = "text/plain",
        original_filename: str | None = None,
    ) -> SourceAddResult:
        name = _required_text(display_name, "display_name")
        text = _required_text(content, "content")
        normalized_media_type = _required_text(media_type, "media_type").lower()
        if normalized_media_type not in self.limits.supported_media_types:
            raise ValueError(f"unsupported media_type: {normalized_media_type}")
        filename = original_filename.strip() if isinstance(original_filename, str) and original_filename.strip() else None
        source_bytes = len(text.encode("utf-8"))
        if source_bytes > self.limits.max_source_bytes:
            raise ValueError("source exceeds max_source_bytes")
        extracted_characters = len(text)
        if extracted_characters > self.limits.max_extracted_characters:
            raise ValueError("source exceeds max_extracted_characters")
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        existing_id = self._hash_to_id.get(content_hash)
        if existing_id is not None:
            return SourceAddResult(source=self._sources[existing_id], created=False)
        if len(self._sources) >= self.limits.max_sources:
            raise ValueError("session reached max_sources")
        source = UploadedSource(
            source_id=f"upload-{content_hash[:16]}",
            display_name=name,
            content=text,
            content_hash=content_hash,
            source_type=SourceType.USER_UPLOAD,
            created_at=self._clock(),
            extraction=ExtractionMetadata(
                media_type=normalized_media_type,
                original_filename=filename,
                source_bytes=source_bytes,
                extracted_characters=extracted_characters,
            ),
        )
        self._sources[source.source_id] = source
        self._hash_to_id[content_hash] = source.source_id
        return SourceAddResult(source=source, created=True)

    def list_sources(self) -> tuple[UploadedSource, ...]:
        """Return the sources currently registered in this session."""

        return tuple(self._sources.values())

    def get_source(self, source_id: str) -> UploadedSource:
        identifier = _required_text(source_id, "source_id")
        try:
            return self._sources[identifier]
        except KeyError as exc:
            raise SourceNotFoundError(f"uploaded source not found: {identifier}") from exc

    def delete_source(self, source_id: str) -> UploadedSource:
        source = self.get_source(source_id)
        del self._sources[source.source_id]
        del self._hash_to_id[source.content_hash]
        return source
