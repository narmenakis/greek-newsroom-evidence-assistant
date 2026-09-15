"""Shared value objects used by dense, lexical, and fused retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass(frozen=True)
class RetrievedChunk:
    """A retrieved chunk with its rank and relevance score."""

    document: Document
    score: float
    rank: int
