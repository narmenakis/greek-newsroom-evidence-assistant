import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from langchain_core.documents import Document

from journalism_rag.reranker import CrossEncoderReranker, format_qwen3_pair
from journalism_rag.retrieval_types import RetrievedChunk


def chunk(chunk_id: str, text: str, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        document=Document(page_content=text, metadata={"chunk_id": chunk_id}),
        score=0.1,
        rank=rank,
    )


class FakeScorer:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def predict(self, sentences, *, batch_size, show_progress_bar, convert_to_numpy):
        self.calls.append(
            (sentences, batch_size, show_progress_bar, convert_to_numpy)
        )
        return self.scores


class Phase3RerankerTests(unittest.TestCase):
    def test_qwen_pair_format_is_explicit_and_deterministic(self):
        self.assertEqual(
            format_qwen3_pair("Ποιος;", "Το άρθρο."),
            "<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n"
            "<Query>: Ποιος;\n<Document>: Το άρθρο.",
        )

    def test_reranker_sorts_candidates_and_refreshes_ranks(self):
        scorer = FakeScorer([0.2, 0.9, 0.4])
        reranker = CrossEncoderReranker(model=scorer, batch_size=7)
        candidates = [
            chunk("a", "first article chunk", 1),
            chunk("b", "second article chunk", 2),
            chunk("c", "third article chunk", 3),
        ]

        results = reranker.rerank("question", candidates, limit=2)

        self.assertEqual(
            [result.document.metadata["chunk_id"] for result in results], ["b", "c"]
        )
        self.assertEqual([result.score for result in results], [0.9, 0.4])
        self.assertEqual([result.rank for result in results], [1, 2])
        self.assertEqual(
            scorer.calls,
            [
                (
                    [
                        ("question", "first article chunk"),
                        ("question", "second article chunk"),
                        ("question", "third article chunk"),
                    ],
                    7,
                    False,
                    True,
                )
            ],
        )

    def test_empty_candidates_do_not_call_model(self):
        scorer = FakeScorer([])
        reranker = CrossEncoderReranker(model=scorer)
        self.assertEqual(reranker.rerank("question", []), [])
        self.assertEqual(scorer.calls, [])

    def test_invalid_inputs_and_score_count_fail_clearly(self):
        scorer = FakeScorer([0.2])
        reranker = CrossEncoderReranker(model=scorer)
        candidates = [chunk("a", "article", 1), chunk("b", "article", 2)]

        with self.assertRaises(ValueError):
            reranker.rerank("", candidates)
        with self.assertRaises(ValueError):
            reranker.rerank("question", candidates, limit=0)
        with self.assertRaisesRegex(ValueError, "different number"):
            reranker.rerank("question", candidates)


if __name__ == "__main__":
    unittest.main()
