"""Rank-fusion utilities for combining independent retrieval systems."""

from __future__ import annotations

from collections import OrderedDict
from typing import Sequence

from .retrieval_types import RetrievedChunk


def _result_key(result: RetrievedChunk) -> str:
    metadata = result.document.metadata
    if metadata.get("chunk_id"):
        return str(metadata["chunk_id"])
    if metadata.get("article_id") is not None or metadata.get("chunk_index") is not None:
        return f"{metadata.get('article_id', '')}:{metadata.get('chunk_index', '')}"
    return result.document.page_content


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[RetrievedChunk]],
    *,
    limit: int = 5,
    fusion_constant: int = 60,
) -> list[RetrievedChunk]:
    """Merge rankings using RRF score = sum(1 / (constant + rank))."""

    if limit <= 0:
        raise ValueError("limit must be greater than zero")
    if fusion_constant < 0:
        raise ValueError("fusion_constant cannot be negative")
    fused: OrderedDict[str, dict] = OrderedDict()
    for results in result_lists:
        for result in results:
            key = _result_key(result)
            entry = fused.setdefault(key, {"score": 0.0, "document": result.document})
            entry["score"] += 1.0 / (fusion_constant + result.rank)
    ranked = sorted(fused.values(), key=lambda entry: entry["score"], reverse=True)[:limit]
    return [
        RetrievedChunk(document=entry["document"], score=entry["score"], rank=rank)
        for rank, entry in enumerate(ranked, start=1)
    ]
