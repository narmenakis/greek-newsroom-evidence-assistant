"""Session-local searchable index for uploaded sources."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from ..chunker import chunk_documents
from ..config import Settings
from ..lexical import BM25Retriever
from ..retrieval_types import RetrievedChunk
from .models import InMemorySourceStore, SourceAddResult, SourceNotFoundError, UploadedSource


class WhitespaceTokenizer:
    """Small deterministic tokenizer for the session-local lexical index."""

    def __init__(self) -> None:
        self._tokens: dict[int, str] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        ids: list[int] = []
        for word in text.split():
            token_id = len(self._tokens)
            self._tokens[token_id] = word
            ids.append(token_id)
        return ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(self._tokens[token_id] for token_id in token_ids)


class UploadedSourceIndex:
    """Chunk and search uploaded sources without touching the corpus index.

    The index is rebuilt after each source mutation because the session-sized
    upload set is intentionally small.  A later vector-backed implementation
    can preserve this public contract while replacing the BM25 implementation.
    """

    def __init__(
        self,
        tokenizer: Any | None = None,
        *,
        settings: Settings | None = None,
        max_chunks_per_source: int = 2_048,
    ) -> None:
        self.settings = settings or Settings()
        self.settings.validate()
        if max_chunks_per_source <= 0:
            raise ValueError("max_chunks_per_source must be greater than zero")
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.max_chunks_per_source = max_chunks_per_source
        self._chunks_by_source: dict[str, tuple[Document, ...]] = {}
        self._retriever: BM25Retriever | None = None

    def add_source(self, source: UploadedSource) -> int:
        """Index one source and return its generated chunk count."""

        if source.source_id in self._chunks_by_source:
            return len(self._chunks_by_source[source.source_id])
        document = Document(
            page_content=source.content,
            metadata={
                "article_id": source.source_id,
                "source_id": source.source_id,
                "source_type": source.source_type.value,
                "title": source.display_name,
                "datetime": source.created_at.timestamp(),
            },
        )
        chunks = tuple(
            chunk_documents(
                [document],
                self.tokenizer,
                chunk_size_tokens=self.settings.chunk_size_tokens,
                chunk_overlap_tokens=self.settings.chunk_overlap_tokens,
            )
        )
        if not chunks:
            raise ValueError(f"source produced no indexable chunks: {source.source_id}")
        if len(chunks) > self.max_chunks_per_source:
            raise ValueError(f"source exceeds max_chunks_per_source: {source.source_id}")
        self._chunks_by_source[source.source_id] = chunks
        self._rebuild()
        return len(chunks)

    def delete_source(self, source_id: str) -> int:
        """Remove one source and its indexed chunks."""

        try:
            chunks = self._chunks_by_source.pop(source_id)
        except KeyError as exc:
            raise SourceNotFoundError(f"uploaded source not indexed: {source_id}") from exc
        self._rebuild()
        return len(chunks)

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve only from the currently indexed uploaded sources."""

        if not query or not query.strip():
            raise ValueError("query cannot be empty")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        if self._retriever is None:
            return []
        return self._retriever.retrieve(query, limit=limit, where=where)

    def indexed_source_ids(self) -> tuple[str, ...]:
        """Return source IDs currently represented in the upload index."""

        return tuple(self._chunks_by_source)

    def _rebuild(self) -> None:
        documents = [chunk for chunks in self._chunks_by_source.values() for chunk in chunks]
        self._retriever = BM25Retriever(self.settings, documents=documents) if documents else None


class IndexedSourceCatalog:
    """Keep the source registry and upload index consistent."""

    def __init__(self, store: InMemorySourceStore, index: UploadedSourceIndex) -> None:
        self.store = store
        self.index = index

    def add_source(self, *args: Any, **kwargs: Any) -> SourceAddResult:
        result = self.store.add_source(*args, **kwargs)
        if not result.created:
            return result
        try:
            self.index.add_source(result.source)
        except Exception:
            self.store.delete_source(result.source.source_id)
            raise
        return result

    def list_sources(self) -> tuple[UploadedSource, ...]:
        return self.store.list_sources()

    def get_source(self, source_id: str) -> UploadedSource:
        return self.store.get_source(source_id)

    def delete_source(self, source_id: str) -> UploadedSource:
        source = self.store.get_source(source_id)
        self.index.delete_source(source.source_id)
        return self.store.delete_source(source.source_id)
