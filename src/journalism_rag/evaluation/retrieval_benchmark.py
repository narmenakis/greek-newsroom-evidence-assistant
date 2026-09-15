"""Provider-free retrieval evaluation for the versioned journalism fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..config import Settings
from ..loader import canonicalize_url, load_corpus
from ..retrieval import HybridRetriever
from .fixtures import load_question_fixture, validate_labels_against_documents
from .reranker_benchmark import metadata_filter, ranked_article_urls, retrieval_query


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case_metrics(
    ranked_urls: Sequence[str], question: Mapping[str, Any]
) -> dict[str, float | int | bool | None]:
    """Calculate article-level metrics for one case.

    Recall is the fraction of the case's relevant articles found in the cutoff.
    Hit rate records whether any relevant article was found. Unanswerable cases
    return null ranking metrics and are excluded from aggregate denominators.
    """

    if not question.get("answerable", True):
        return {
            "answerable": False,
            "relevant_article_count": 0,
            "recall_at_5": None,
            "recall_at_10": None,
            "hit_at_5": None,
            "hit_at_10": None,
            "mrr": None,
            "ndcg_at_10": None,
        }

    relevant = {canonicalize_url(url) for url in question.get("relevant_urls", [])}
    ranked = [canonicalize_url(url) for url in ranked_urls]
    first_relevant_rank = next(
        (rank for rank, url in enumerate(ranked, start=1) if url in relevant),
        None,
    )
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, url in enumerate(ranked[:10], start=1)
        if url in relevant
    )
    ideal_count = min(len(relevant), 10)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))

    def recall_at(cutoff: int) -> float:
        return len(relevant.intersection(ranked[:cutoff])) / len(relevant)

    def hit_at(cutoff: int) -> float:
        return float(bool(relevant.intersection(ranked[:cutoff])))

    return {
        "answerable": True,
        "relevant_article_count": len(relevant),
        "recall_at_5": round(recall_at(5), 6),
        "recall_at_10": round(recall_at(10), 6),
        "hit_at_5": hit_at(5),
        "hit_at_10": hit_at(10),
        "mrr": round(1.0 / first_relevant_rank, 6) if first_relevant_rank else 0.0,
        "ndcg_at_10": round(dcg / ideal_dcg, 6) if ideal_dcg else 0.0,
    }


def aggregate_metrics(case_results: Sequence[Mapping[str, Any]]) -> dict[str, float | int | None]:
    """Aggregate metrics over answerable cases only."""

    answerable = [case for case in case_results if case.get("answerable", True)]
    metric_names = (
        "recall_at_5",
        "recall_at_10",
        "hit_at_5",
        "hit_at_10",
        "mrr",
        "ndcg_at_10",
    )
    aggregate: dict[str, float | int | None] = {
        "answerable_cases": len(answerable),
        "excluded_no_answer_cases": len(case_results) - len(answerable),
    }
    for metric_name in metric_names:
        values = [float(case[metric_name]) for case in answerable if case[metric_name] is not None]
        aggregate[metric_name] = round(sum(values) / len(values), 6) if values else None
    return aggregate


def evaluate_retriever(
    questions: Sequence[Mapping[str, Any]],
    retriever: Any,
    *,
    retrieval_limit: int = 10,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Run one retriever and return per-case results plus aggregate metrics."""

    if retrieval_limit <= 0:
        raise ValueError("retrieval_limit must be greater than zero")
    cases: list[dict[str, Any]] = []
    for question in questions:
        query = retrieval_query(dict(question))
        where = metadata_filter(dict(question))
        started = clock()
        results = retriever.retrieve(query, limit=retrieval_limit, where=where)
        elapsed = clock() - started
        urls = ranked_article_urls(results)
        metrics = case_metrics(urls, question)
        cases.append(
            {
                "id": question["id"],
                "language": question["language"],
                "category": question["category"],
                "task_type": question["task_type"],
                "topic": question["topic"],
                "difficulty": question["difficulty"],
                "answerable": question["answerable"],
                "retrieval_query": query,
                "metadata_filter": where,
                "relevant_urls": list(question["relevant_urls"]),
                "retrieved_urls": urls[:retrieval_limit],
                "result_chunk_count": len(results),
                "unique_article_count": len(urls),
                "retrieval_seconds": round(elapsed, 6),
                **metrics,
            }
        )
    return {
        "cases": cases,
        "metrics": aggregate_metrics(cases),
        "retrieval_seconds": round(
            sum(float(case["retrieval_seconds"]) for case in cases), 6
        ),
    }


def _group_values(
    question: Mapping[str, Any],
    dimension: str,
    corpus_metadata: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    if dimension == "topic":
        return [str(question["topic"])]
    if dimension == "language":
        return [str(question["language"])]
    if dimension == "filter_behavior":
        return ["filtered" if question.get("filters") else "unfiltered"]
    if dimension == "outlet":
        outlets = {
            str(corpus_metadata[canonicalize_url(url)].get("website", "unknown"))
            for url in question.get("relevant_urls", [])
            if canonicalize_url(url) in corpus_metadata
        }
        return sorted(outlets or {"unknown"})
    raise ValueError(f"unsupported breakdown dimension: {dimension}")


def breakdown_metrics(
    questions: Sequence[Mapping[str, Any]],
    case_results: Sequence[Mapping[str, Any]],
    corpus_metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    """Group answerable case metrics by topic, outlet, language, and filters."""

    by_id = {str(case["id"]): case for case in case_results}
    breakdowns: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for dimension in ("topic", "outlet", "language", "filter_behavior"):
        groups: dict[str, list[Mapping[str, Any]]] = {}
        for question in questions:
            if not question.get("answerable", True):
                continue
            case = by_id[str(question["id"])]
            for value in _group_values(question, dimension, corpus_metadata):
                groups.setdefault(value, []).append(case)
        breakdowns[dimension] = {
            value: aggregate_metrics(group_cases)
            for value, group_cases in sorted(groups.items())
        }
    return breakdowns


def run_benchmark(
    settings: Settings,
    questions: Sequence[Mapping[str, Any]],
    *,
    retrieval_limit: int = 10,
) -> dict[str, Any]:
    """Run dense and hybrid retrieval against the configured local index."""

    documents = load_corpus(settings.corpus_path)
    label_validation = validate_labels_against_documents(list(questions), documents)
    corpus_metadata = {
        canonicalize_url(str(document.metadata["url"])): document.metadata
        for document in documents
    }

    hybrid = HybridRetriever(settings)
    retrievers = {"dense": hybrid.dense, "hybrid": hybrid}
    systems: dict[str, Any] = {}
    for name, retriever in retrievers.items():
        result = evaluate_retriever(
            questions,
            retriever,
            retrieval_limit=retrieval_limit,
        )
        result["breakdowns"] = breakdown_metrics(
            questions, result["cases"], corpus_metadata
        )
        systems[name] = result

    return {
        "benchmark_id": "tempi-retrieval-benchmark-v1",
        "evaluation_type": "offline_retrieval_only",
        "question_count": len(questions),
        "answerable_count": sum(bool(question["answerable"]) for question in questions),
        "no_answer_count": sum(not question["answerable"] for question in questions),
        "retrieval_limit": retrieval_limit,
        "corpus": {
            "path": str(settings.corpus_path),
            "collection": settings.collection_name,
            **label_validation,
        },
        "systems": systems,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate dense and hybrid retrieval offline")
    parser.add_argument(
        "--questions", type=Path, default=Path("evaluation/tempi_questions.jsonl")
    )
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--chroma-path", type=Path)
    parser.add_argument("--retrieval-limit", type=int, default=10)
    parser.add_argument(
        "--output", type=Path, default=Path("evaluation/retrieval_benchmark.json")
    )
    args = parser.parse_args()
    if args.retrieval_limit <= 0:
        raise SystemExit("--retrieval-limit must be greater than zero")

    settings = Settings.from_env()
    settings = replace(
        settings,
        corpus_path=args.corpus or settings.corpus_path,
        chroma_path=args.chroma_path or settings.chroma_path,
    )
    questions = load_question_fixture(args.questions)
    report = run_benchmark(
        settings,
        questions,
        retrieval_limit=args.retrieval_limit,
    )
    report["questions"] = {
        "path": str(args.questions),
        "sha256": _sha256_file(args.questions),
    }
    report["corpus"]["sha256"] = _sha256_file(settings.corpus_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
