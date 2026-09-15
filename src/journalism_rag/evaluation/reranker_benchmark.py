"""Compare hybrid retrieval with optional cross-encoder reranking."""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ..config import Settings
from ..loader import canonicalize_url
from ..reranker import CrossEncoderReranker, QWEN3_RERANKER_MODEL, Qwen3PairScorer
from ..retrieval import HybridRetriever
from ..retrieval_types import RetrievedChunk


def load_questions(path: Path) -> list[dict[str, Any]]:
    """Load the JSONL fixture and reject malformed or duplicate case IDs."""

    questions: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            question = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number}") from exc
        case_id = question.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"line {line_number} is missing a non-empty id")
        if case_id in seen_ids:
            raise ValueError(f"duplicate evaluation case id: {case_id}")
        if not isinstance(question.get("question"), str) or not question["question"].strip():
            raise ValueError(f"case {case_id} is missing a question")
        seen_ids.add(case_id)
        questions.append(question)
    if not questions:
        raise ValueError("evaluation fixture is empty")
    return questions


def retrieval_query(question: dict[str, Any]) -> str:
    """Make a deterministic standalone query for benchmark-only follow-ups."""

    current = question["question"].strip()
    context = question.get("conversation_context")
    if isinstance(context, str) and context.strip():
        return f"{context.strip()}\n{current}"
    return current


def metadata_filter(question: dict[str, Any]) -> dict[str, Any] | None:
    """Translate the fixture's compact filter format to retriever syntax."""

    filters = question.get("filters")
    if not filters:
        return None
    where: dict[str, Any] = {}
    for field, values in filters.items():
        if field == "website":
            if isinstance(values, list):
                values = [_normalise_website(value) for value in values]
            elif isinstance(values, str):
                values = _normalise_website(values)
        if isinstance(values, list):
            where[field] = {"$in": values}
        else:
            where[field] = values
    return where


def _normalise_website(value: object) -> str:
    """Match URL-style ``www.`` labels to the corpus website metadata."""

    website = str(value).strip().lower()
    return website[4:] if website.startswith("www.") else website


def ranked_article_urls(results: Sequence[RetrievedChunk]) -> list[str]:
    """Return unique article URLs in chunk ranking order."""

    urls: list[str] = []
    seen: set[str] = set()
    for result in results:
        url = result.document.metadata.get("url")
        if not url:
            continue
        normalized = canonicalize_url(str(url))
        if normalized not in seen:
            seen.add(normalized)
            urls.append(normalized)
    return urls


def diversity_select(
    candidates: Sequence[RetrievedChunk], *, limit: int
) -> list[RetrievedChunk]:
    """Select high-scoring evidence while spreading results across outlets.

    This is an evaluation-only heuristic, not a learned model: a new article
    is preferred over a duplicate, then an outlet with fewer selections, then
    the original retrieval score.  Measuring it separately prevents a simple
    diversity preference from being mistaken for a reranker quality gain.
    """

    if limit <= 0:
        raise ValueError("limit must be greater than zero")
    remaining = list(candidates)
    selected: list[RetrievedChunk] = []
    seen_urls: set[str] = set()
    outlet_counts: dict[str, int] = {}
    while remaining and len(selected) < limit:
        def key(candidate: RetrievedChunk) -> tuple[int, int, float, int]:
            url = canonicalize_url(str(candidate.document.metadata.get("url", "")))
            outlet = str(candidate.document.metadata.get("website", "")).lower()
            return (
                int(url in seen_urls),
                outlet_counts.get(outlet, 0),
                -float(candidate.score),
                candidate.rank,
            )

        chosen = min(remaining, key=key)
        remaining.remove(chosen)
        selected.append(chosen)
        seen_urls.add(canonicalize_url(str(chosen.document.metadata.get("url", ""))))
        outlet = str(chosen.document.metadata.get("website", "")).lower()
        outlet_counts[outlet] = outlet_counts.get(outlet, 0) + 1
    return [
        RetrievedChunk(document=item.document, score=item.score, rank=rank)
        for rank, item in enumerate(selected, start=1)
    ]


def ranking_metrics(
    ranked_urls_by_case: Sequence[Sequence[str]],
    questions: Sequence[dict[str, Any]],
) -> dict[str, float | int | None]:
    """Calculate URL-level Recall@5/10, MRR, and nDCG@10."""

    answerable = [question for question in questions if question.get("answerable", True)]
    if not answerable:
        return {
            "answerable_cases": 0,
            "recall_at_5": None,
            "recall_at_10": None,
            "mrr": None,
            "ndcg_at_10": None,
        }

    recalls = {5: [], 10: []}
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    for index, question in enumerate(questions):
        if not question.get("answerable", True):
            continue
        relevant = {
            canonicalize_url(url) for url in question.get("relevant_urls", [])
        }
        ranked = [canonicalize_url(url) for url in ranked_urls_by_case[index]]
        for k in recalls:
            recalls[k].append(
                len(relevant.intersection(ranked[:k])) / len(relevant)
                if relevant
                else 0.0
            )
        first = next(
            (rank for rank, url in enumerate(ranked, start=1) if url in relevant),
            None,
        )
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


def _peak_memory_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024) if sys.platform == "darwin" else value / 1024


def benchmark_questions(
    questions: Sequence[dict[str, Any]],
    retriever: Any,
    reranker: Any,
    *,
    candidate_limit: int = 20,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Measure baseline and reranked retrieval for each fixture case."""

    if candidate_limit <= 0:
        raise ValueError("candidate_limit must be greater than zero")
    baseline_urls: list[list[str]] = []
    reranked_urls: list[list[str]] = []
    diverse_urls: list[list[str]] = []
    cases: list[dict[str, Any]] = []
    for question in questions:
        query = retrieval_query(question)
        where = metadata_filter(question)
        started = clock()
        candidates = retriever.retrieve(query, limit=candidate_limit, where=where)
        retrieval_elapsed = clock() - started
        started = clock()
        reranked = reranker.rerank(query, candidates, limit=candidate_limit)
        rerank_elapsed = clock() - started
        diverse = diversity_select(candidates, limit=candidate_limit)
        baseline = ranked_article_urls(candidates)
        reranked_urls_for_case = ranked_article_urls(reranked)
        baseline_urls.append(baseline)
        reranked_urls.append(reranked_urls_for_case)
        diverse_urls.append(ranked_article_urls(diverse))
        cases.append(
            {
                "id": question["id"],
                "retrieval_query": query,
                "candidate_count": len(candidates),
                "metadata_filter": where,
                "retrieval_seconds": round(retrieval_elapsed, 6),
                "reranker_seconds": round(rerank_elapsed, 6),
                "total_with_reranker_seconds": round(
                    retrieval_elapsed + rerank_elapsed, 6
                ),
                "baseline_urls": baseline[:10],
                "reranked_urls": reranked_urls_for_case[:10],
                "diversity_urls": diverse_urls[-1][:10],
            }
        )
    baseline_time = sum(case["retrieval_seconds"] for case in cases)
    reranker_time = sum(case["reranker_seconds"] for case in cases)
    return {
        "cases": cases,
        "baseline": {
            "metrics": ranking_metrics(baseline_urls, questions),
            "retrieval_seconds": round(baseline_time, 6),
        },
        "reranked": {
            "metrics": ranking_metrics(reranked_urls, questions),
            "retrieval_seconds": round(baseline_time, 6),
            "reranker_seconds": round(reranker_time, 6),
            "total_seconds": round(baseline_time + reranker_time, 6),
        },
        "diversity": {
            "metrics": ranking_metrics(diverse_urls, questions),
        },
    }


def run_benchmark(
    settings: Settings,
    questions: Sequence[dict[str, Any]],
    *,
    candidate_limit: int = 20,
    reranker_model: str = "BAAI/bge-reranker-v2-m3",
    reranker_batch_size: int = 16,
    reranker_max_length: int = 256,
    reranker_device: str = "cpu",
) -> dict[str, Any]:
    """Run the local benchmark with a real hybrid retriever and reranker."""

    started = time.perf_counter()
    retriever = HybridRetriever(settings)
    retriever_ready_seconds = time.perf_counter() - started
    started = time.perf_counter()
    if reranker_model == QWEN3_RERANKER_MODEL:
        scorer = Qwen3PairScorer(
            reranker_model,
            device=reranker_device,
            max_length=reranker_max_length,
            batch_size=reranker_batch_size,
            local_files_only=True,
        )
        reranker = CrossEncoderReranker(
            reranker_model,
            device=reranker_device,
            max_length=reranker_max_length,
            batch_size=reranker_batch_size,
            model=scorer,
        )
    else:
        reranker = CrossEncoderReranker(
            reranker_model,
            device=reranker_device,
            max_length=reranker_max_length,
            batch_size=reranker_batch_size,
            local_files_only=True,
        )
    reranker_load_seconds = time.perf_counter() - started
    report = benchmark_questions(
        questions,
        retriever,
        reranker,
        candidate_limit=candidate_limit,
    )
    report.update(
        {
            "candidate_limit": candidate_limit,
            "retriever_ready_seconds": round(retriever_ready_seconds, 3),
            "reranker_load_seconds": round(reranker_load_seconds, 3),
            "reranker_model": reranker_model,
            "reranker_device": reranker_device,
            "reranker_batch_size": reranker_batch_size,
            "reranker_max_length": reranker_max_length,
            "peak_memory_mb": round(_peak_memory_mb(), 2),
        }
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark hybrid retrieval with reranking")
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("evaluation/tempi_questions.jsonl"),
    )
    parser.add_argument("--candidate-limit", type=int, default=20)
    parser.add_argument(
        "--reranker-model",
        default="BAAI/bge-reranker-v2-m3",
        help="Hugging Face reranker ID; Qwen3-Reranker uses its causal-LM adapter",
    )
    parser.add_argument(
        "--reranker-device",
        choices=("cpu", "mps"),
        default="cpu",
    )
    parser.add_argument("--reranker-batch-size", type=int, default=16)
    parser.add_argument("--reranker-max-length", type=int, default=256)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/reranker_benchmark.json"),
    )
    args = parser.parse_args()
    if args.candidate_limit <= 0:
        raise SystemExit("--candidate-limit must be greater than zero")
    if args.reranker_batch_size <= 0:
        raise SystemExit("--reranker-batch-size must be greater than zero")
    if args.reranker_max_length <= 0:
        raise SystemExit("--reranker-max-length must be greater than zero")

    settings = Settings.from_env()
    questions = load_questions(args.questions)
    report = run_benchmark(
        settings,
        questions,
        candidate_limit=args.candidate_limit,
        reranker_model=args.reranker_model,
        reranker_batch_size=args.reranker_batch_size,
        reranker_max_length=args.reranker_max_length,
        reranker_device=args.reranker_device,
    )
    output = {
        "benchmark_id": "tempi-reranker-benchmark-v1",
        "corpus": str(settings.corpus_path),
        "questions": str(args.questions),
        "question_count": len(questions),
        **report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
