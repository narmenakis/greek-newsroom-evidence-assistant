"""Load and validate versioned offline evaluation fixtures.

The fixture loader checks the evaluation contract without loading an embedding
model, vector store, or language model.  Labels remain provisional until a
human reviews them against the scraped article text.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from ..loader import canonicalize_url


SUPPORTED_LANGUAGES = {"el", "en"}
SUPPORTED_CATEGORIES = {
    "fact_retrieval",
    "synthesis",
    "timeline",
    "conflicting_reports",
    "follow_up",
    "metadata_filter",
    "negative_control",
    "no_answer",
}
SUPPORTED_TASK_TYPES = {
    "fact_lookup",
    "synthesis",
    "timeline",
    "temporal_comparison",
    "follow_up",
    "filtered_retrieval",
    "abstention",
}
SUPPORTED_DIFFICULTIES = {"easy", "medium", "hard"}
SUPPORTED_REVIEW_STATUSES = {
    "assistant_reviewed_pending_human_confirmation",
    "human_confirmed",
    "needs_revision",
}
_FILTER_FIELDS = {"author", "website", "section", "datetime"}
_ARTICLE_ID_PATTERN = re.compile(r"^[0-9a-f]{16}$")


def _non_empty_string(value: Any, field: str, case_id: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"case {case_id} field {field!r} must be a non-empty string")
    return value.strip()


def _validate_filters(filters: Any, case_id: str) -> None:
    if filters is None:
        return
    if not isinstance(filters, dict):
        raise ValueError(f"case {case_id} filters must be an object")
    unknown = set(filters) - _FILTER_FIELDS
    if unknown:
        raise ValueError(f"case {case_id} has unsupported filter fields: {sorted(unknown)}")
    for field, values in filters.items():
        if isinstance(values, list):
            if not values or not all(isinstance(value, str) and value.strip() for value in values):
                raise ValueError(f"case {case_id} filter {field!r} must contain non-empty strings")
        elif not isinstance(values, (str, int, float)) or (isinstance(values, str) and not values.strip()):
            raise ValueError(f"case {case_id} filter {field!r} has an invalid value")


def _validate_case(case: Any, line_number: int, seen_ids: set[str]) -> dict[str, Any]:
    if not isinstance(case, dict):
        raise ValueError(f"line {line_number} must contain a JSON object")
    case_id = _non_empty_string(case.get("id"), "id", f"line {line_number}")
    if case_id in seen_ids:
        raise ValueError(f"duplicate evaluation case id: {case_id}")
    seen_ids.add(case_id)

    language = _non_empty_string(case.get("language"), "language", case_id)
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"case {case_id} has unsupported language: {language!r}")
    category = _non_empty_string(case.get("category"), "category", case_id)
    if category not in SUPPORTED_CATEGORIES:
        raise ValueError(f"case {case_id} has unsupported category: {category!r}")
    task_type = _non_empty_string(case.get("task_type"), "task_type", case_id)
    if task_type not in SUPPORTED_TASK_TYPES:
        raise ValueError(f"case {case_id} has unsupported task type: {task_type!r}")
    _non_empty_string(case.get("topic"), "topic", case_id)
    difficulty = _non_empty_string(case.get("difficulty"), "difficulty", case_id)
    if difficulty not in SUPPORTED_DIFFICULTIES:
        raise ValueError(f"case {case_id} has unsupported difficulty: {difficulty!r}")
    _non_empty_string(case.get("question"), "question", case_id)

    answerable = case.get("answerable", True)
    if not isinstance(answerable, bool):
        raise ValueError(f"case {case_id} field 'answerable' must be boolean")
    urls = case.get("relevant_urls", [])
    if not isinstance(urls, list):
        raise ValueError(f"case {case_id} relevant_urls must be an array")
    normalized_urls = [canonicalize_url(_non_empty_string(url, "relevant_urls", case_id)) for url in urls]
    if len(normalized_urls) != len(set(normalized_urls)):
        raise ValueError(f"case {case_id} contains duplicate relevant URLs")
    if not answerable and normalized_urls:
        raise ValueError(f"unanswerable case {case_id} cannot have relevant URLs")
    if answerable and not normalized_urls:
        raise ValueError(f"answerable case {case_id} must have at least one relevant URL")

    article_ids = case.get("relevant_article_ids")
    if not isinstance(article_ids, list) or not all(
        isinstance(article_id, str) and _ARTICLE_ID_PATTERN.fullmatch(article_id)
        for article_id in article_ids
    ):
        raise ValueError(f"case {case_id} relevant_article_ids must contain lowercase 16-digit hex IDs")
    if len(article_ids) != len(normalized_urls):
        raise ValueError(f"case {case_id} must pair every relevant URL with one article ID")
    if len(article_ids) != len(set(article_ids)):
        raise ValueError(f"case {case_id} contains duplicate relevant article IDs")

    acceptable_evidence = case.get("acceptable_evidence")
    if not isinstance(acceptable_evidence, list) or not all(
        isinstance(note, str) and note.strip() for note in acceptable_evidence
    ):
        raise ValueError(f"case {case_id} acceptable_evidence must be an array of strings")
    if answerable and not acceptable_evidence:
        raise ValueError(f"answerable case {case_id} must describe acceptable evidence")
    if not answerable and acceptable_evidence:
        raise ValueError(f"unanswerable case {case_id} cannot describe acceptable evidence")

    review = case.get("label_review")
    if not isinstance(review, dict):
        raise ValueError(f"case {case_id} label_review must be an object")
    status = _non_empty_string(review.get("status"), "label_review.status", case_id)
    if status not in SUPPORTED_REVIEW_STATUSES:
        raise ValueError(f"case {case_id} has unsupported review status: {status!r}")
    _non_empty_string(review.get("basis"), "label_review.basis", case_id)
    _non_empty_string(review.get("notes"), "label_review.notes", case_id)

    context = case.get("conversation_context")
    if context is not None:
        _non_empty_string(context, "conversation_context", case_id)
    _validate_filters(case.get("filters"), case_id)
    return case


def load_question_fixture(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL question fixture and enforce its versioned contract."""

    fixture_path = Path(path)
    if not fixture_path.is_file():
        raise ValueError(f"evaluation fixture does not exist: {fixture_path}")
    questions: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(fixture_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number}") from exc
        questions.append(_validate_case(case, line_number, seen_ids))
    if not questions:
        raise ValueError(f"evaluation fixture is empty: {fixture_path}")
    return questions


def fixture_summary(questions: list[dict[str, Any]]) -> dict[str, Any]:
    """Return deterministic counts suitable for a validation smoke check."""

    return {
        "question_count": len(questions),
        "answerable_count": sum(bool(question.get("answerable", True)) for question in questions),
        "unanswerable_count": sum(not question.get("answerable", True) for question in questions),
        "categories": sorted({question["category"] for question in questions}),
        "task_types": sorted({question["task_type"] for question in questions}),
        "difficulties": sorted({question["difficulty"] for question in questions}),
        "languages": sorted({question["language"] for question in questions}),
        "review_statuses": sorted(
            {question["label_review"]["status"] for question in questions}
        ),
    }


def validate_labels_against_documents(
    questions: list[dict[str, Any]], documents: list[Any]
) -> dict[str, int]:
    """Verify that every labeled URL and article ID identifies one corpus document.

    This is an identity check, not a semantic relevance judgment. The review notes
    in the fixture record the separate content review.
    """

    documents_by_url: dict[str, Any] = {}
    for document in documents:
        metadata = getattr(document, "metadata", None)
        if not isinstance(metadata, dict):
            raise ValueError("corpus document must expose a metadata object")
        url = canonicalize_url(_non_empty_string(metadata.get("url"), "url", "corpus document"))
        if url in documents_by_url:
            raise ValueError(f"corpus contains duplicate canonical URL: {url}")
        documents_by_url[url] = document

    label_pair_count = 0
    for question in questions:
        for raw_url, expected_article_id in zip(
            question["relevant_urls"], question["relevant_article_ids"], strict=True
        ):
            url = canonicalize_url(raw_url)
            document = documents_by_url.get(url)
            if document is None:
                raise ValueError(f"case {question['id']} labels URL absent from corpus: {url}")
            actual_article_id = document.metadata.get("article_id")
            if actual_article_id != expected_article_id:
                raise ValueError(
                    f"case {question['id']} article ID mismatch for {url}: "
                    f"expected {expected_article_id}, corpus has {actual_article_id}"
                )
            label_pair_count += 1

    return {
        "corpus_document_count": len(documents_by_url),
        "validated_label_pair_count": label_pair_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an offline journalism evaluation fixture")
    parser.add_argument("path", type=Path, default=Path("evaluation/tempi_questions.jsonl"), nargs="?")
    parser.add_argument(
        "--corpus",
        type=Path,
        help="also verify every labeled URL/article-ID pair against this local corpus CSV",
    )
    args = parser.parse_args()
    questions = load_question_fixture(args.path)
    summary = fixture_summary(questions)
    if args.corpus:
        from ..loader import load_corpus

        summary.update(validate_labels_against_documents(questions, load_corpus(args.corpus)))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
