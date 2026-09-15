import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from langchain_core.documents import Document

from journalism_rag.config import Settings
from journalism_rag.llm import ChatModel, GenerationResult
from journalism_rag.pipeline import AnswerStatus, GroundedRAG
from journalism_rag.providers.llm import DeepSeekChatModel
from journalism_rag.retrieval import RetrievedChunk


class FakeRetriever:
    def retrieve(self, query, *, limit=5, where=None):
        return [
            RetrievedChunk(
                document=Document(
                    page_content="Το άρθρο αναφέρει 57 νεκρούς.",
                    metadata={
                        "article_id": "article_1",
                        "chunk_id": "chunk_1",
                        "url": "https://example.com/tempi",
                        "title": "Τέμπη",
                    },
                ),
                score=0.9,
                rank=1,
            )
        ]


class RecordingDirectModel(ChatModel):
    provider = "fake"
    model = "fake-direct-model"

    def __init__(self):
        self.calls = []

    def generate(self, messages, options=None, *, tools=None, tool_history=None):
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "tool_history": tool_history,
            }
        )
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


class FakeResponse:
    status_code = 200
    text = ""

    def json(self):
        return {
            "choices": [
                {"message": {"content": "Direct answer"}, "finish_reason": "stop"}
            ]
        }


class FakeSession:
    def __init__(self):
        self.calls = []

    def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return FakeResponse()


class Phase6DirectRAGTests(unittest.TestCase):
    def test_grounded_rag_answers_without_tools_or_agent_history(self):
        model = RecordingDirectModel()

        result = GroundedRAG(
            Settings(),
            retriever=FakeRetriever(),
            chat_model=model,
        ).answer("Πόσοι νεκροί αναφέρονται;", limit=1)

        self.assertEqual(result.status, AnswerStatus.ANSWERED)
        self.assertTrue(result.citation_validation.valid)
        self.assertEqual(result.used_source_ids, ("S1",))
        self.assertEqual(len(model.calls), 1)
        self.assertIsNone(model.calls[0]["tools"])
        self.assertIsNone(model.calls[0]["tool_history"])

    def test_direct_deepseek_payload_exposes_no_tools(self):
        session = FakeSession()

        DeepSeekChatModel(
            Settings(deepseek_api_key="fake-secret"), session=session
        ).generate([{"role": "user", "content": "Answer directly"}])

        payload = session.calls[0][1]["json"]
        self.assertNotIn("tools", payload)
        self.assertTrue(
            all(message.get("role") != "tool" for message in payload["messages"])
        )


if __name__ == "__main__":
    unittest.main()
