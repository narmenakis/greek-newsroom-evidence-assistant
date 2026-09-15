import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from langchain_core.documents import Document

from journalism_rag.citations import (
    ENGLISH_ABSTENTION,
    GREEK_ABSTENTION,
    validate_citations,
)
from journalism_rag.config import Settings
from journalism_rag.llm import ChatModel, GenerationResult
from journalism_rag.pipeline import AnswerStatus, GroundedRAG
from journalism_rag.retrieval import RetrievedChunk


class StaticRetriever:
    def __init__(self, results):
        self.results = results

    def retrieve(self, query, *, limit=5, where=None):
        return self.results[:limit]


class StaticModel(ChatModel):
    provider = "fake"
    model = "fake-model"

    def __init__(self, text):
        self.text = text
        self.called = False

    def generate(self, messages, options=None):
        self.called = True
        return GenerationResult(
            text=self.text,
            provider=self.provider,
            model=self.model,
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=8,
            total_tokens=18,
            latency_seconds=0.01,
        )


def evidence():
    return [
        RetrievedChunk(
            document=Document(
                page_content="Το άρθρο αναφέρει 57 νεκρούς.",
                metadata={
                    "article_id": "a1",
                    "chunk_id": "a1:c1:0",
                    "url": "https://example.com/a",
                    "title": "Τέμπη",
                    "author": "Reporter",
                    "website": "example.com",
                    "section": "News",
                    "datetime": 1.0,
                },
            ),
            score=0.91,
            rank=1,
        )
    ]


class Phase3GroundingTests(unittest.TestCase):
    def test_validates_claim_citations_and_post_punctuation_label(self):
        result = validate_citations(
            "Αναφέρονται 57 νεκροί. [S1] Η ενημέρωση έγινε την Παρασκευή [S2].",
            {"S1", "S2"},
        )
        self.assertTrue(result.valid)
        self.assertEqual(result.cited_source_ids, ("S1", "S2"))

    def test_rejects_nonexistent_source_id(self):
        result = validate_citations("Αναφέρονται 57 νεκροί [S9].", {"S1"})
        self.assertFalse(result.valid)
        self.assertEqual(result.invalid_source_ids, ("S9",))

    def test_allows_uncited_claim_as_coverage_warning_when_valid_citation_exists(self):
        result = validate_citations(
            "Αναφέρονται 57 νεκροί. Η ενημέρωση έγινε την Παρασκευή [S1].",
            {"S1"},
        )
        self.assertTrue(result.valid)
        self.assertTrue(result.coverage_warning)
        self.assertEqual(result.uncited_claims, ("Αναφέρονται 57 νεκροί.",))

    def test_rejects_answer_without_any_citation(self):
        result = validate_citations("Αναφέρονται 57 νεκροί.", {"S1"})
        self.assertFalse(result.valid)
        self.assertEqual(result.uncited_claims, ("Αναφέρονται 57 νεκροί.",))

    def test_no_retrieval_results_abstains_without_calling_model(self):
        model = StaticModel("should not be used")
        result = GroundedRAG(
            Settings(),
            retriever=StaticRetriever([]),
            chat_model=model,
        ).answer("What caused the accident?")
        self.assertEqual(result.status, AnswerStatus.ABSTAINED)
        self.assertEqual(result.answer, ENGLISH_ABSTENTION)
        self.assertEqual(result.abstention_reason, "no_evidence_retrieved")
        self.assertFalse(model.called)
        self.assertIsNone(result.generation)

    def test_invalid_generation_fails_closed_to_greek_abstention(self):
        result = GroundedRAG(
            Settings(),
            retriever=StaticRetriever(evidence()),
            chat_model=StaticModel("Αναφέρονται 57 νεκροί [S9]."),
        ).answer("Πόσοι νεκροί αναφέρονται;")
        self.assertEqual(result.status, AnswerStatus.ABSTAINED)
        self.assertEqual(result.answer, GREEK_ABSTENTION)
        self.assertEqual(result.abstention_reason, "citation_validation_failed")
        self.assertEqual(result.citation_validation.invalid_source_ids, ("S9",))
        self.assertIsNotNone(result.generation)

    def test_model_abstention_is_normalized_to_query_language(self):
        result = GroundedRAG(
            Settings(),
            retriever=StaticRetriever(evidence()),
            chat_model=StaticModel(GREEK_ABSTENTION),
        ).answer("What is not covered?")
        self.assertEqual(result.status, AnswerStatus.ABSTAINED)
        self.assertEqual(result.answer, ENGLISH_ABSTENTION)
        self.assertEqual(result.abstention_reason, "model_abstained")


if __name__ == "__main__":
    unittest.main()
