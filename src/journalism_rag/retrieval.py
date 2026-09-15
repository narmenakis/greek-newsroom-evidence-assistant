"""Dense and hybrid retrieval services for the journalism RAG pipeline."""

from __future__ import annotations

from typing import Any

from langchain_chroma import Chroma

from .config import Settings
from .indexer import INDEXER_COMMAND, IndexContractError, _embedding_function, _validate_collection
from .retrieval_types import RetrievedChunk
from .fusion import reciprocal_rank_fusion
from .lexical import BM25Retriever


class DenseRetriever:
    """Open a configured collection and return ranked source chunks."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()
        self.settings.validate()
        self._db = Chroma(
            collection_name=self.settings.collection_name,
            persist_directory=str(self.settings.chroma_path),
            embedding_function=_embedding_function(self.settings),
        )
        collection_ids = {str(item) for item in self._db.get().get("ids", [])}
        if not collection_ids:
            raise IndexContractError(
                f"Chroma collection {self.settings.collection_name!r} is missing or empty. "
                f"Build it with `{INDEXER_COMMAND}` before starting retrieval."
            )
        _validate_collection(self._db, self.settings)

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        if not query or not query.strip():
            raise ValueError("query cannot be empty")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        results = self._db.similarity_search_with_relevance_scores(
            query.strip(), k=limit, filter=where
        )
        return [
            RetrievedChunk(document=document, score=float(score), rank=rank)
            for rank, (document, score) in enumerate(results, start=1)
        ]


class HybridRetriever:
    """Combine dense E5 and lexical BM25 rankings with reciprocal-rank fusion."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()
        self.dense = DenseRetriever(self.settings)
        self.lexical = BM25Retriever(self.settings)

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        if not query or not query.strip():
            raise ValueError("query cannot be empty")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        # Keep enough candidates for fusion even when the caller asks for a
        # small final evidence set (the Phase 3 default is 30 from each path).
        candidate_limit = max(limit * 5, 30)
        dense_results = self.dense.retrieve(query, limit=candidate_limit, where=where)
        lexical_results = self.lexical.retrieve(query, limit=candidate_limit, where=where)
        return reciprocal_rank_fusion(
            [dense_results, lexical_results],
            limit=limit,
        )
