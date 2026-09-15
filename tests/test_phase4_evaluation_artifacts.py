import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from langchain_core.documents import Document

from journalism_rag.evaluation.fixtures import (
    fixture_summary,
    load_question_fixture,
    validate_labels_against_documents,
)
from journalism_rag.loader import canonicalize_url


ROOT = Path(__file__).parents[1]


def _case(case_id: str = "case-1", *, answerable: bool = True) -> dict:
    urls = ["https://example.com/article/"] if answerable else []
    article_ids = ["0123456789abcdef"] if answerable else []
    evidence = ["A directly reported fact."] if answerable else []
    return {
        "id": case_id,
        "language": "el",
        "category": "fact_retrieval" if answerable else "no_answer",
        "task_type": "fact_lookup" if answerable else "abstention",
        "topic": "test_topic",
        "difficulty": "easy",
        "question": "Test question?",
        "answerable": answerable,
        "relevant_urls": urls,
        "relevant_article_ids": article_ids,
        "acceptable_evidence": evidence,
        "label_review": {
            "status": "assistant_reviewed_pending_human_confirmation",
            "basis": "focused unit-test fixture",
            "notes": "Synthetic metadata; no model is called.",
        },
    }


class Phase4EvaluationArtifactTests(unittest.TestCase):
    def test_tempi_fixture_is_versioned_and_has_expected_summary(self):
        questions = load_question_fixture(ROOT / "evaluation/tempi_questions.jsonl")
        self.assertEqual(
            fixture_summary(questions),
            {
                "question_count": 11,
                "answerable_count": 10,
                "unanswerable_count": 1,
                "categories": [
                    "conflicting_reports",
                    "fact_retrieval",
                    "follow_up",
                    "metadata_filter",
                    "negative_control",
                    "no_answer",
                    "synthesis",
                    "timeline",
                ],
                "task_types": [
                    "abstention",
                    "fact_lookup",
                    "filtered_retrieval",
                    "follow_up",
                    "synthesis",
                    "temporal_comparison",
                    "timeline",
                ],
                "difficulties": ["easy", "hard", "medium"],
                "languages": ["el"],
                "review_statuses": ["human_confirmed"],
            },
        )
        manifest = json.loads((ROOT / "evaluation/evaluation_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], "tempi-questions-v1")
        self.assertEqual(manifest["questions"]["question_count"], len(questions))
        self.assertEqual(
            manifest["questions"]["labeled_url_article_id_pairs"],
            sum(len(question["relevant_article_ids"]) for question in questions),
        )
        fixture_hash = hashlib.sha256(
            (ROOT / "evaluation/tempi_questions.jsonl").read_bytes()
        ).hexdigest()
        self.assertEqual(manifest["questions"]["sha256"], fixture_hash)

    def test_loader_rejects_duplicate_ids_and_inconsistent_answerability(self):
        duplicate_a = _case("same")
        duplicate_b = _case("same")
        inconsistent = _case("noanswer", answerable=False)
        inconsistent["relevant_urls"] = ["https://example.com/article/"]
        inconsistent["relevant_article_ids"] = ["0123456789abcdef"]
        cases = [
            (
                "\n".join((json.dumps(duplicate_a), json.dumps(duplicate_b))),
                "duplicate",
            ),
            (
                json.dumps(inconsistent),
                "unanswerable",
            ),
        ]
        for content, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "questions.jsonl"
                path.write_text(content, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    load_question_fixture(path)

    def test_corpus_validation_checks_url_and_stable_article_id_pair(self):
        questions = [_case()]
        document = Document(
            page_content="Synthetic article body.",
            metadata={
                "url": canonicalize_url("https://example.com/article/"),
                "article_id": "0123456789abcdef",
            },
        )
        self.assertEqual(
            validate_labels_against_documents(questions, [document]),
            {"corpus_document_count": 1, "validated_label_pair_count": 1},
        )

        questions[0]["relevant_article_ids"] = ["fedcba9876543210"]
        with self.assertRaisesRegex(ValueError, "article ID mismatch"):
            validate_labels_against_documents(questions, [document])

    def test_human_review_packet_covers_every_case_and_labeled_article(self):
        questions = load_question_fixture(ROOT / "evaluation/tempi_questions.jsonl")
        packet = (ROOT / "evaluation/label_review.md").read_text(encoding="utf-8")

        for question in questions:
            with self.subTest(case_id=question["id"]):
                self.assertIn(f"`{question['id']}`", packet)
                for article_id in question["relevant_article_ids"]:
                    self.assertIn(f"`{article_id}`", packet)


if __name__ == "__main__":
    unittest.main()
