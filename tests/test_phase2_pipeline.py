import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from langchain_core.documents import Document

from journalism_rag.config import Settings
from journalism_rag.llm import ChatModel, GenerationResult, GenerationStreamEvent
from journalism_rag.pipeline import AnswerStatus, GroundedRAG
from journalism_rag.retrieval import RetrievedChunk


class FakeRetriever:
    def retrieve(self, query, *, limit=5, where=None):
        return [
            RetrievedChunk(
                document=Document(
                    page_content="Το άρθρο αναφέρει 57 νεκρούς.",
                    metadata={"article_id": "a1", "url": "https://example.com/a", "title": "Τέμπη"},
                ),
                score=0.91,
                rank=1,
            )
        ]


class FakeChatModel(ChatModel):
    provider = "fake"
    model = "fake-model"

    def __init__(self):
        self.messages = None

    def generate(self, messages, options=None):
        self.messages = messages
        return GenerationResult(
            text="Αναφέρονται 57 νεκροί [S1].",
            provider=self.provider,
            model=self.model,
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=8,
            total_tokens=18,
            latency_seconds=0.01,
        )


class FakeStreamingChatModel(FakeChatModel):
    supports_streaming = True

    def stream(self, messages, options=None, **kwargs):
        del messages, options, kwargs
        yield GenerationStreamEvent.delta("Αναφέρονται ")
        yield GenerationStreamEvent.delta("57 νεκροί [S1].")
        yield GenerationStreamEvent.completed(
            GenerationResult(
                text="Αναφέρονται 57 νεκροί [S1].",
                provider=self.provider,
                model=self.model,
                finish_reason="stop",
                prompt_tokens=10,
                completion_tokens=8,
                total_tokens=18,
                latency_seconds=0.01,
            )
        )


class InvalidStreamingChatModel(FakeStreamingChatModel):
    def stream(self, messages, options=None, **kwargs):
        del messages, options, kwargs
        yield GenerationStreamEvent.delta("Ισχυρισμός χωρίς έγκυρη πηγή [S9].")
        yield GenerationStreamEvent.completed(
            GenerationResult(
                text="Ισχυρισμός χωρίς έγκυρη πηγή [S9].",
                provider=self.provider,
                model=self.model,
                finish_reason="stop",
                prompt_tokens=10,
                completion_tokens=8,
                total_tokens=18,
                latency_seconds=0.01,
            )
        )


class FakeReranker:
    def __init__(self):
        self.calls = []

    def rerank(self, query, candidates, *, limit):
        self.calls.append((query, len(candidates), limit))
        return list(candidates)[:limit]


class Phase2PipelineTests(unittest.TestCase):
    def test_grounded_answer_contains_labeled_evidence_and_provenance(self):
        model = FakeChatModel()
        result = GroundedRAG(
            Settings(),
            retriever=FakeRetriever(),
            chat_model=model,
        ).answer("Πόσοι νεκροί αναφέρονται;", limit=1)
        self.assertEqual(result.status, AnswerStatus.ANSWERED)
        self.assertEqual(result.used_source_ids, ("S1",))
        self.assertEqual(result.sources[0].url, "https://example.com/a")
        self.assertEqual(result.sources[0].text, "Το άρθρο αναφέρει 57 νεκρούς.")
        self.assertTrue(result.citation_validation.valid)
        self.assertIn("[S1]", model.messages[1]["content"])
        self.assertIn("μόνο τα αποσπάσματα", model.messages[0]["content"])
        self.assertIn("αναφέρουν διαφορετικούς αριθμούς", model.messages[0]["content"])
        self.assertIn("do not silently", model.messages[0]["content"])

    def test_opt_in_reranker_uses_candidate_and_evidence_limits(self):
        model = FakeChatModel()
        reranker = FakeReranker()
        settings = Settings(
            reranker_candidate_limit=20,
            reranker_evidence_limit=5,
        )
        rag = GroundedRAG(
            settings,
            retriever=FakeRetriever(),
            chat_model=model,
            reranker=reranker,
        )
        result = rag.answer("Πόσοι νεκροί αναφέρονται;", limit=3)
        self.assertEqual(result.status, AnswerStatus.ANSWERED)
        self.assertEqual(reranker.calls, [("Πόσοι νεκροί αναφέρονται;", 1, 3)])

    def test_stream_answer_updates_provisionally_then_validates_final_text(self):
        updates = []
        result = GroundedRAG(
            Settings(),
            retriever=FakeRetriever(),
            chat_model=FakeStreamingChatModel(),
        ).stream_answer(
            "Πόσοι νεκροί αναφέρονται?",
            limit=1,
            on_update=updates.append,
        )
        self.assertEqual(
            updates,
            ["Αναφέρονται ", "Αναφέρονται 57 νεκροί [S1]."],
        )
        self.assertEqual(result.status, AnswerStatus.ANSWERED)
        self.assertTrue(result.citation_validation.valid)

    def test_stream_answer_does_not_accept_invalid_citations(self):
        result = GroundedRAG(
            Settings(),
            retriever=FakeRetriever(),
            chat_model=InvalidStreamingChatModel(),
        ).stream_answer("Πόσοι νεκροί αναφέρονται?", limit=1)
        self.assertEqual(result.status, AnswerStatus.ABSTAINED)
        self.assertEqual(result.abstention_reason, "citation_validation_failed")
        self.assertNotIn("[S9]", result.answer)


if __name__ == "__main__":
    unittest.main()
