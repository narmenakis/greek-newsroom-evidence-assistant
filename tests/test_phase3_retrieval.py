import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from langchain_core.documents import Document

from journalism_rag.config import Settings
from journalism_rag.fusion import reciprocal_rank_fusion
from journalism_rag.lexical import BM25Retriever, matches_where, tokenize
from journalism_rag.retrieval import HybridRetriever, RetrievedChunk


def chunk(chunk_id: str, text: str, **metadata) -> RetrievedChunk:
    return RetrievedChunk(
        document=Document(
            page_content=text,
            metadata={"chunk_id": chunk_id, **metadata},
        ),
        score=0.5,
        rank=1,
    )


class Phase3RetrievalTests(unittest.TestCase):
    def test_tokenize_is_case_and_accent_insensitive_for_greek(self):
        self.assertEqual(tokenize("Τέμπη — ΤΕΜΠΗ, 57 νεκροί"), ["τεμπη", "τεμπη", "57", "νεκροι"])

    def test_where_filter_supports_sidebar_operators(self):
        metadata = {"website": "Outlet A", "datetime": 10.0, "section": "News"}
        where = {
            "$and": [
                {"website": {"$in": ["Outlet A", "Outlet B"]}},
                {"datetime": {"$gte": 5.0, "$lte": 10.0}},
            ]
        }
        self.assertTrue(matches_where(metadata, where))
        self.assertFalse(matches_where(metadata, {"section": "Opinion"}))

    def test_bm25_returns_term_matching_chunks_and_applies_filters(self):
        documents = [
            Document(
                page_content="Η επιτροπή ερευνά τα Τέμπη",
                metadata={
                    "chunk_id": "a",
                    "author": "Reporter",
                    "website": "kathimerini.gr",
                    "section": "News",
                    "datetime": 10.0,
                },
            ),
            Document(
                page_content="Η επιτροπή ερευνά τα Τέμπη",
                metadata={
                    "chunk_id": "b",
                    "author": "Other Reporter",
                    "website": "kathimerini.gr",
                    "section": "News",
                    "datetime": 10.0,
                },
            ),
            Document(
                page_content="Η επιτροπή ερευνά τα Τέμπη",
                metadata={
                    "chunk_id": "c",
                    "author": "Reporter",
                    "website": "other.gr",
                    "section": "News",
                    "datetime": 10.0,
                },
            ),
            Document(
                page_content="Η πρόγνωση του καιρού",
                metadata={"chunk_id": "d"},
            ),
        ]
        retriever = BM25Retriever(Settings(), documents=documents)
        results = retriever.retrieve("ΤΕΜΠΗ")
        self.assertEqual(
            [result.document.metadata["chunk_id"] for result in results], ["a", "b", "c"]
        )
        where = {
            "$and": [
                {"author": {"$in": ["Reporter"]}},
                {"website": {"$in": ["kathimerini.gr"]}},
                {"section": {"$in": ["News"]}},
                {"datetime": {"$gte": 10.0, "$lte": 10.0}},
            ]
        }
        filtered = retriever.retrieve("ΤΕΜΠΗ", where=where)
        self.assertEqual([result.document.metadata["chunk_id"] for result in filtered], ["a"])
        self.assertEqual(retriever.retrieve("ΤΕΜΠΗ", where={"chunk_id": "d"}), [])

    def test_rrf_rewards_documents_present_in_both_rankings(self):
        dense = [chunk("shared", "shared"), chunk("dense-only", "dense")]
        lexical = [chunk("shared", "shared"), chunk("lexical-only", "lexical")]
        results = reciprocal_rank_fusion([dense, lexical], limit=3)
        self.assertEqual([result.document.metadata["chunk_id"] for result in results], ["shared", "dense-only", "lexical-only"])
        self.assertGreater(results[0].score, results[1].score)
        self.assertEqual([result.rank for result in results], [1, 2, 3])

    def test_rrf_rejects_invalid_limits(self):
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([], limit=0)
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([], fusion_constant=-1)

    def test_hybrid_retriever_fuses_bounded_candidate_lists(self):
        class FakeSource:
            def __init__(self, results):
                self.results = results
                self.calls = []

            def retrieve(self, query, *, limit, where):
                self.calls.append((query, limit, where))
                return self.results

        dense = FakeSource([chunk("dense", "dense")])
        lexical = FakeSource([chunk("lexical", "lexical")])
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.dense = dense
        retriever.lexical = lexical

        results = retriever.retrieve("question", limit=3, where={"section": "News"})

        self.assertEqual([result.document.metadata["chunk_id"] for result in results], ["dense", "lexical"])
        self.assertEqual(dense.calls, [("question", 30, {"section": "News"})])
        self.assertEqual(lexical.calls, [("question", 30, {"section": "News"})])


if __name__ == "__main__":
    unittest.main()
