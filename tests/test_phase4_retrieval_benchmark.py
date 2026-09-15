import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from langchain_core.documents import Document

from journalism_rag.evaluation.retrieval_benchmark import (
    aggregate_metrics,
    breakdown_metrics,
    case_metrics,
    evaluate_retriever,
)
from journalism_rag.retrieval_types import RetrievedChunk


def chunk(url: str, rank: int, website: str = "kathimerini.gr") -> RetrievedChunk:
    return RetrievedChunk(
        document=Document(
            page_content=f"article {url}",
            metadata={"url": url, "chunk_id": f"{url}:{rank}", "website": website},
        ),
        score=1.0 / rank,
        rank=rank,
    )


class FakeRetriever:
    def __init__(self, results_by_query):
        self.results_by_query = results_by_query
        self.calls = []

    def retrieve(self, query, *, limit, where):
        self.calls.append((query, limit, where))
        return self.results_by_query[query]


class Phase4RetrievalBenchmarkTests(unittest.TestCase):
    def test_case_metrics_use_fractional_recall_and_null_for_no_answer(self):
        question = {
            "answerable": True,
            "relevant_urls": [
                "https://example.test/one/",
                "https://example.test/two/",
            ],
        }
        metrics = case_metrics(
            [
                "https://example.test/irrelevant",
                "https://example.test/one",
            ],
            question,
        )
        self.assertEqual(metrics["recall_at_5"], 0.5)
        self.assertEqual(metrics["hit_at_5"], 1.0)
        self.assertEqual(metrics["mrr"], 0.5)

        no_answer = case_metrics([], {"answerable": False, "relevant_urls": []})
        self.assertIsNone(no_answer["recall_at_10"])
        self.assertEqual(aggregate_metrics([metrics, no_answer])["answerable_cases"], 1)
        self.assertEqual(aggregate_metrics([metrics, no_answer])["excluded_no_answer_cases"], 1)

    def test_evaluation_preserves_followup_filter_and_breakdowns(self):
        questions = [
            {
                "id": "fact",
                "language": "el",
                "category": "fact_retrieval",
                "task_type": "fact_lookup",
                "topic": "casualties",
                "difficulty": "easy",
                "question": "Πόσοι;",
                "answerable": True,
                "relevant_urls": ["https://example.test/one/"],
                "filters": {"website": ["www.kathimerini.gr"]},
            },
            {
                "id": "followup",
                "language": "el",
                "category": "follow_up",
                "task_type": "follow_up",
                "topic": "investigation",
                "difficulty": "medium",
                "question": "Ποια από αυτά;",
                "conversation_context": "Τι συνέβη;",
                "answerable": True,
                "relevant_urls": ["https://example.test/two/"],
            },
            {
                "id": "none",
                "language": "el",
                "category": "no_answer",
                "task_type": "abstention",
                "topic": "outside",
                "difficulty": "easy",
                "question": "Δεν υπάρχει;",
                "answerable": False,
                "relevant_urls": [],
            },
        ]
        retriever = FakeRetriever(
            {
                "Πόσοι;": [chunk("https://example.test/one", 1)],
                "Τι συνέβη;\nΠοια από αυτά;": [chunk("https://example.test/two", 1)],
                "Δεν υπάρχει;": [],
            }
        )
        result = evaluate_retriever(
            questions,
            retriever,
            retrieval_limit=5,
            clock=iter([0.0, 0.1, 1.0, 1.2, 2.0, 2.05]).__next__,
        )
        self.assertEqual(result["metrics"]["answerable_cases"], 2)
        self.assertEqual(result["metrics"]["excluded_no_answer_cases"], 1)
        self.assertEqual(retriever.calls[0], ("Πόσοι;", 5, {"website": {"$in": ["kathimerini.gr"]}}))
        self.assertEqual(retriever.calls[1][0], "Τι συνέβη;\nΠοια από αυτά;")
        self.assertEqual(
            breakdown_metrics(
                questions,
                result["cases"],
                {
                    "https://example.test/one": {"website": "kathimerini.gr"},
                    "https://example.test/two": {"website": "skai.gr"},
                },
            )["outlet"]["skai.gr"]["answerable_cases"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
