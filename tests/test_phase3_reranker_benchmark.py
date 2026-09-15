import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from langchain_core.documents import Document

from journalism_rag.evaluation.reranker_benchmark import (
    benchmark_questions,
    diversity_select,
    load_questions,
    metadata_filter,
    ranking_metrics,
    retrieval_query,
)
from journalism_rag.evaluation.embedding_benchmark import _metrics as embedding_metrics
from journalism_rag.retrieval_types import RetrievedChunk


def chunk(url: str, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        document=Document(
            page_content=f"article {url}",
            metadata={"url": url, "chunk_id": f"{url}:{rank}"},
        ),
        score=1.0 / rank,
        rank=rank,
    )


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def retrieve(self, query, *, limit, where):
        self.calls.append((query, limit, where))
        return [chunk("https://example.test/irrelevant", 1), chunk("https://example.test/relevant", 2)]


class FakeReranker:
    def rerank(self, query, candidates, *, limit):
        return [candidates[1], candidates[0]][:limit]


class Phase3RerankerBenchmarkTests(unittest.TestCase):
    def test_embedding_and_reranker_metrics_use_fractional_recall(self):
        questions = [
            {
                "answerable": True,
                "relevant_urls": [
                    "https://example.test/one/",
                    "https://example.test/two/",
                    "https://example.test/three/",
                ],
            }
        ]
        ranked = [[
            "https://example.test/one",
            "https://example.test/two",
            "https://example.test/irrelevant",
        ]]

        self.assertEqual(
            embedding_metrics(ranked, questions)["recall_at_5"],
            0.666667,
        )
        self.assertEqual(
            ranking_metrics(ranked, questions)["recall_at_5"],
            0.666667,
        )

    def test_diversity_selector_prefers_new_outlet_and_article(self):
        candidates = [
            RetrievedChunk(
                document=Document(
                    page_content="a",
                    metadata={"url": "https://example.test/a", "website": "same"},
                ),
                score=1.0,
                rank=1,
            ),
            RetrievedChunk(
                document=Document(
                    page_content="a duplicate",
                    metadata={"url": "https://example.test/a", "website": "other"},
                ),
                score=0.99,
                rank=2,
            ),
            RetrievedChunk(
                document=Document(
                    page_content="b",
                    metadata={"url": "https://example.test/b", "website": "other"},
                ),
                score=0.5,
                rank=3,
            ),
        ]
        selected = diversity_select(candidates, limit=2)
        self.assertEqual(
            [item.document.metadata["url"] for item in selected],
            ["https://example.test/a", "https://example.test/b"],
        )

    def test_fixture_helpers_preserve_followup_context_and_filters(self):
        question = {
            "id": "followup",
            "question": "Ποια από αυτά;",
            "conversation_context": "Τι συνέβη στα Τέμπη;",
            "filters": {"website": ["www.kathimerini.gr"]},
        }
        self.assertEqual(retrieval_query(question), "Τι συνέβη στα Τέμπη;\nΠοια από αυτά;")
        self.assertEqual(
            metadata_filter(question), {"website": {"$in": ["kathimerini.gr"]}}
        )

    def test_metrics_and_benchmark_show_reranking_gain_and_timing(self):
        questions = [
            {
                "id": "case-1",
                "question": "question",
                "answerable": True,
                "relevant_urls": ["https://example.test/relevant/"],
            }
        ]
        retriever = FakeRetriever()
        report = benchmark_questions(
            questions,
            retriever,
            FakeReranker(),
            candidate_limit=2,
            clock=iter([0.0, 0.2, 1.0, 1.4]).__next__,
        )

        self.assertEqual(report["baseline"]["metrics"]["mrr"], 0.5)
        self.assertEqual(report["reranked"]["metrics"]["mrr"], 1.0)
        self.assertEqual(report["baseline"]["retrieval_seconds"], 0.2)
        self.assertEqual(report["reranked"]["reranker_seconds"], 0.4)
        self.assertEqual(retriever.calls[0][1], 2)

    def test_load_questions_rejects_duplicate_ids(self, tmp_path=None):
        # Use a temporary file without requiring a fixture or model.
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "questions.jsonl"
            path.write_text(
                '{"id":"same","question":"one"}\n{"id":"same","question":"two"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_questions(path)


if __name__ == "__main__":
    unittest.main()
