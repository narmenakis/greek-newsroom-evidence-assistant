"""Deterministic BM25 retrieval over the same token-bounded corpus chunks."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Any, Iterable

from langchain_core.documents import Document
from transformers import AutoTokenizer

from .chunker import chunk_documents
from .config import Settings
from .loader import load_corpus


_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Normalize Greek/Latin text while retaining words and numeric terms."""

    normalized = unicodedata.normalize("NFD", text.casefold())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return _TOKEN_PATTERN.findall(normalized)


def matches_where(metadata: dict[str, Any], where: dict[str, Any] | None) -> bool:
    """Evaluate the Chroma filter subset used by the archive sidebar."""

    if not where:
        return True
    if "$and" in where:
        return all(matches_where(metadata, clause) for clause in where["$and"])
    if "$or" in where:
        return any(matches_where(metadata, clause) for clause in where["$or"])
    for field, condition in where.items():
        value = metadata.get(field)
        if isinstance(condition, dict):
            for operator, expected in condition.items():
                if operator == "$in" and value not in expected:
                    return False
                if operator == "$gte" and (value is None or value < expected):
                    return False
                if operator == "$lte" and (value is None or value > expected):
                    return False
        elif value != condition:
            return False
    return True


class BM25Index:
    """Small in-memory BM25 index suitable for the current 518-chunk corpus."""

    def __init__(self, tokenized_documents: Iterable[list[str]], *, k1: float = 1.5, b: float = 0.75):
        self.documents = tuple(tuple(tokens) for tokens in tokenized_documents)
        if not self.documents:
            raise ValueError("BM25 index cannot be empty")
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 > 0 and 0 <= b <= 1")
        self.k1 = k1
        self.b = b
        self.document_lengths = tuple(len(tokens) for tokens in self.documents)
        self.average_length = sum(self.document_lengths) / len(self.documents)
        document_frequency: Counter[str] = Counter()
        self.term_frequencies = tuple(Counter(tokens) for tokens in self.documents)
        for frequencies in self.term_frequencies:
            document_frequency.update(frequencies.keys())
        self.inverse_document_frequency = {
            term: math.log(1 + (len(self.documents) - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def scores(self, query_tokens: list[str]) -> list[float]:
        if not query_tokens:
            return [0.0] * len(self.documents)
        query_terms = set(query_tokens)
        scores: list[float] = []
        for frequencies, document_length in zip(
            self.term_frequencies, self.document_lengths, strict=True
        ):
            score = 0.0
            normalization = self.k1 * (1 - self.b + self.b * document_length / self.average_length)
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                idf = self.inverse_document_frequency[term]
                score += idf * (frequency * (self.k1 + 1)) / (frequency + normalization)
            scores.append(score)
        return scores


class BM25Retriever:
    """Retrieve corpus chunks by lexical BM25 relevance."""

    def __init__(self, settings: Settings | None = None, *, documents: list[Document] | None = None):
        self.settings = settings or Settings.from_env()
        self.settings.validate()
        if documents is None:
            tokenizer = AutoTokenizer.from_pretrained(self.settings.embedding_model)
            documents = chunk_documents(
                load_corpus(self.settings.corpus_path),
                tokenizer,
                chunk_size_tokens=self.settings.chunk_size_tokens,
                chunk_overlap_tokens=self.settings.chunk_overlap_tokens,
            )
        if not documents:
            raise ValueError("BM25 corpus cannot be empty")
        self.documents = tuple(documents)
        self._index = BM25Index(tokenize(document.page_content) for document in self.documents)

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[Any]:
        if not query or not query.strip():
            raise ValueError("query cannot be empty")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        query_tokens = tokenize(query)
        scores = self._index.scores(query_tokens)
        ranked = sorted(
            (
                (score, document)
                for score, document in zip(scores, self.documents, strict=True)
                if score > 0 and matches_where(document.metadata, where)
            ),
            key=lambda item: item[0],
            reverse=True,
        )[:limit]
        from .retrieval_types import RetrievedChunk

        return [
            RetrievedChunk(document=document, score=float(score), rank=rank)
            for rank, (score, document) in enumerate(ranked, start=1)
        ]
