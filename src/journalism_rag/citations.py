"""Citation coverage and abstention rules for grounded answers."""

from __future__ import annotations

import re
from dataclasses import dataclass


GREEK_ABSTENTION = "Δεν υπάρχουν επαρκή στοιχεία στο διαθέσιμο corpus."
ENGLISH_ABSTENTION = "There is insufficient evidence in the available corpus."

_CITATION_PATTERN = re.compile(r"\[([SCUW]\d+)\]")
_GREEK_PATTERN = re.compile(r"[Α-Ωα-ωΆΈΉΊΌΎΏάέήίόύώϊϋΐΰ]")
_WORD_PATTERN = re.compile(r"[\wΆ-ώ]+", re.UNICODE)


@dataclass(frozen=True)
class CitationValidation:
    """Machine-readable result of structural citation validation."""

    valid: bool
    abstained: bool
    cited_source_ids: tuple[str, ...]
    invalid_source_ids: tuple[str, ...]
    uncited_claims: tuple[str, ...]
    coverage_warning: bool = False


def abstention_for_query(query: str) -> str:
    """Return the fixed abstention sentence in the query language."""

    return GREEK_ABSTENTION if _GREEK_PATTERN.search(query) else ENGLISH_ABSTENTION


def is_exact_abstention(answer: str) -> bool:
    normalized = answer.strip().strip('"\'«»').casefold()
    return normalized in {GREEK_ABSTENTION.casefold(), ENGLISH_ABSTENTION.casefold()}


def _ordered_unique(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _claim_segments(answer: str) -> list[str]:
    # Models occasionally place a citation immediately after punctuation. Move
    # it before the boundary so it remains attached to the preceding sentence.
    normalized = re.sub(
        r"([.!?;·])\s*((?:\[[SCUW]\d+\]\s*)+)",
        lambda match: f" {match.group(2).strip()}{match.group(1)} ",
        answer,
    )
    segments: list[str] = []
    for line in normalized.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if not line:
            continue
        segments.extend(
            segment.strip()
            for segment in re.split(r"(?<=[.!?;·])\s+", line)
            if segment.strip()
        )
    return segments


def _is_substantive_claim(segment: str) -> bool:
    plain = _CITATION_PATTERN.sub("", segment).strip()
    if not plain or plain.endswith(":"):
        return False
    return len(_WORD_PATTERN.findall(plain)) >= 3


def validate_citations(answer: str, valid_source_ids: set[str]) -> CitationValidation:
    """Validate citation IDs and report, but do not enforce, sentence coverage.

    This is structural validation only. Semantic claim/evidence entailment is
    intentionally out of product scope and must not be inferred from a valid
    source label.
    """

    if is_exact_abstention(answer):
        return CitationValidation(
            valid=True,
            abstained=True,
            cited_source_ids=(),
            invalid_source_ids=(),
            uncited_claims=(),
            coverage_warning=False,
        )

    cited = _ordered_unique(_CITATION_PATTERN.findall(answer))
    invalid = tuple(source_id for source_id in cited if source_id not in valid_source_ids)
    uncited = tuple(
        segment
        for segment in _claim_segments(answer)
        if _is_substantive_claim(segment) and not _CITATION_PATTERN.search(segment)
    )
    valid_cited = tuple(source_id for source_id in cited if source_id in valid_source_ids)
    return CitationValidation(
        valid=bool(valid_cited) and not invalid,
        abstained=False,
        cited_source_ids=valid_cited,
        invalid_source_ids=invalid,
        uncited_claims=uncited,
        coverage_warning=bool(uncited),
    )
