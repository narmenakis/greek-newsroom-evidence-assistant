"""Benchmark multilingual embedding candidates on the Tempi fixture."""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
from pathlib import Path
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer

from ..chunker import chunk_documents
from ..config import Settings
from ..embeddings import format_document, format_query
from ..loader import canonicalize_url, load_corpus


CANDIDATES = (
    "intfloat/multilingual-e5-large-instruct",
    "intfloat/multilingual-e5-base",
    "Qwen/Qwen3-Embedding-0.6B",
)
def _peak_memory_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB.
    return value / (1024 * 1024) if sys.platform == "darwin" else value / 1024


def _document_texts(model_name: str, chunks) -> list[str]:
    return [format_document(model_name, chunk.page_content) for chunk in chunks]


def _query_text(model_name: str, question: str) -> str:
    return format_query(model_name, question)


def _metrics(ranked_urls: list[list[str]], questions: list[dict]) -> dict[str, float | int | None]:
    answerable = [q for q in questions if q.get("answerable", True)]
    if not answerable:
        return {"answerable_cases": 0, "recall_at_5": None, "recall_at_10": None, "mrr": None, "ndcg_at_10": None}
    recalls = {5: [], 10: []}
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    for index, question in enumerate(questions):
        if not question.get("answerable", True):
            continue
        relevant = {canonicalize_url(url) for url in question.get("relevant_urls", [])}
        ranked = [canonicalize_url(url) for url in ranked_urls[index]]
        for k in recalls:
            recalls[k].append(
                len(relevant.intersection(ranked[:k])) / len(relevant)
                if relevant
                else 0.0
            )
        first = next((rank for rank, url in enumerate(ranked, start=1) if url in relevant), None)
        reciprocal_ranks.append(1.0 / first if first else 0.0)
        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, url in enumerate(ranked[:10], start=1)
            if url in relevant
        )
        ideal_count = min(len(relevant), 10)
        ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
        ndcgs.append(dcg / ideal if ideal else 0.0)
    return {
        "answerable_cases": len(answerable),
        "recall_at_5": round(float(np.mean(recalls[5])), 6),
        "recall_at_10": round(float(np.mean(recalls[10])), 6),
        "mrr": round(float(np.mean(reciprocal_ranks)), 6),
        "ndcg_at_10": round(float(np.mean(ndcgs)), 6),
    }


def benchmark_model(
    model_name: str,
    settings: Settings,
    questions: list[dict],
    batch_size: int,
    *,
    device: str = "cpu",
) -> dict:
    started = time.monotonic()
    result: dict = {"model": model_name, "status": "failed"}
    try:
        documents = load_corpus(settings.corpus_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        chunks = chunk_documents(
            documents,
            tokenizer,
            chunk_size_tokens=settings.chunk_size_tokens,
            chunk_overlap_tokens=settings.chunk_overlap_tokens,
        )
        embedder = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"batch_size": batch_size, "normalize_embeddings": True},
        )
        document_vectors = np.asarray(embedder.embed_documents(_document_texts(model_name, chunks)), dtype=np.float32)
        ranked_urls: list[list[str]] = []
        for question in questions:
            query_vector = np.asarray(
                embedder.embed_query(_query_text(model_name, question["question"])), dtype=np.float32
            )
            scores = document_vectors @ query_vector
            best_by_url: dict[str, float] = {}
            for score, chunk in zip(scores, chunks, strict=True):
                url = chunk.metadata["url"]
                best_by_url[url] = max(float(score), best_by_url.get(url, -1.0))
            ranked_urls.append([url for url, _ in sorted(best_by_url.items(), key=lambda item: item[1], reverse=True)])
        result.update(
            {
                "status": "passed",
                "chunks": len(chunks),
                "metrics": _metrics(ranked_urls, questions),
                "batch_size": batch_size,
                "device": device,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "peak_memory_mb": round(_peak_memory_mb(), 2),
                "ranked_urls": {
                    q["id"]: urls[:10]
                    for q, urls in zip(questions, ranked_urls, strict=True)
                },
            }
        )
    except Exception as exc:  # record candidate failures without hiding other results
        result.update({"error_type": type(exc).__name__, "error": str(exc)[:500]})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark multilingual embedding candidates")
    parser.add_argument("--model", action="append", dest="models", choices=CANDIDATES)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--device",
        choices=("cpu", "mps"),
        default="cpu",
        help="Embedding inference device (use mps on Apple Silicon)",
    )
    parser.add_argument("--output", type=Path, default=Path("evaluation/embedding_benchmark.json"))
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than zero")
    settings = Settings.from_env()
    questions = [
        json.loads(line)
        for line in Path("evaluation/tempi_questions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    models = args.models or list(CANDIDATES)
    report = {
        "benchmark_id": "tempi-embedding-benchmark-v1",
        "corpus": str(settings.corpus_path),
        "questions": "evaluation/tempi_questions.jsonl",
        "chunk_size_tokens": settings.chunk_size_tokens,
        "chunk_overlap_tokens": settings.chunk_overlap_tokens,
        "device": args.device,
        "results": [
            benchmark_model(
                model,
                settings,
                questions,
                args.batch_size,
                device=args.device,
            )
            for model in models
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "results": report["results"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
