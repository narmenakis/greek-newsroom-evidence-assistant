"""Scope-aware retrieval over the permanent corpus and uploaded sources."""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol

from ..fusion import reciprocal_rank_fusion
from ..retrieval_types import RetrievedChunk
from .index import UploadedSourceIndex


class RetrievalScope(str, Enum):
    """The source sets a query is allowed to search."""

    CORPUS_ONLY = "corpus_only"
    UPLOADED_ONLY = "uploaded_only"
    BOTH = "both"


class CorpusRetriever(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        limit: int,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        ...


class ScopedRetriever:
    """Route retrieval to only the corpus, uploads, or both namespaces."""

    def __init__(
        self,
        *,
        corpus_retriever: CorpusRetriever | None,
        uploaded_index: UploadedSourceIndex,
    ) -> None:
        self.corpus_retriever = corpus_retriever
        self.uploaded_index = uploaded_index

    def retrieve(
        self,
        query: str,
        *,
        scope: RetrievalScope | str,
        limit: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        if not query or not query.strip():
            raise ValueError("query cannot be empty")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        try:
            selected_scope = RetrievalScope(scope)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"scope must be one of {[item.value for item in RetrievalScope]}") from exc

        if selected_scope is RetrievalScope.UPLOADED_ONLY:
            return self.uploaded_index.retrieve(query, limit=limit)

        if self.corpus_retriever is None:
            raise RuntimeError("corpus retriever is required for the selected scope")
        if selected_scope is RetrievalScope.CORPUS_ONLY:
            return self.corpus_retriever.retrieve(query, limit=limit, where=where)

        candidate_limit = max(limit * 5, 30)
        corpus_results = self.corpus_retriever.retrieve(query, limit=candidate_limit, where=where)
        uploaded_results = self.uploaded_index.retrieve(query, limit=candidate_limit)
        return reciprocal_rank_fusion([corpus_results, uploaded_results], limit=limit)
