import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from langchain_core.documents import Document

from journalism_rag.config import Settings
from journalism_rag.llm import ChatModel, GenerationResult, LLMRequestError
from journalism_rag.pipeline import AnswerStatus, GroundedRAG
from journalism_rag.query_rewrite import QueryContextualizer
from journalism_rag.retrieval import RetrievedChunk


def generation(text: str) -> GenerationResult:
    return GenerationResult(
        text=text,
        provider="fake",
        model="fake-model",
        finish_reason="stop",
        prompt_tokens=10,
        completion_tokens=8,
        total_tokens=18,
        latency_seconds=0.01,
    )


class SequencedModel(ChatModel):
    provider = "fake"
    model = "fake-model"

    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = []
        self.options = []

    def generate(self, messages, options=None):
        self.messages.append(messages)
        self.options.append(options)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return generation(response)


class RecordingRetriever:
    def __init__(self):
        self.queries = []

    def retrieve(self, query, *, limit=5, where=None):
        self.queries.append((query, limit, where))
        return [
            RetrievedChunk(
                document=Document(
                    page_content="Το άρθρο αναφέρει 57 νεκρούς.",
                    metadata={"article_id": "a1", "chunk_id": "a1:0", "title": "Τέμπη"},
                ),
                score=0.9,
                rank=1,
            )
        ]


class Phase3QueryRewriteTests(unittest.TestCase):
    def test_standalone_question_does_not_call_model(self):
        model = SequencedModel(["unused"])
        contextualizer = QueryContextualizer(model, Settings())
        query = contextualizer.contextualize(
            "Πόσοι νεκροί αναφέρονται στην τραγωδία των Τεμπών;",
            [{"role": "user", "content": "Πες μου για τα Τέμπη."}],
        )
        self.assertEqual(query, "Πόσοι νεκροί αναφέρονται στην τραγωδία των Τεμπών;")
        self.assertEqual(model.messages, [])

    def test_follow_up_is_rewritten_using_recent_history(self):
        model = SequencedModel(["What happened after the reported warnings before the Tempi accident?"])
        contextualizer = QueryContextualizer(model, Settings())
        query = contextualizer.contextualize(
            "What happened afterward?",
            [
                {"role": "user", "content": "What warnings were reported before the Tempi accident?"},
                {"role": "assistant", "content": "The sources describe infrastructure warnings [S1]."},
            ],
        )
        self.assertEqual(query, "What happened after the reported warnings before the Tempi accident?")
        self.assertIn("LATEST QUESTION:\nWhat happened afterward?", model.messages[0][1]["content"])
        self.assertEqual(model.options[0].temperature, 0.0)

    def test_rewrite_failure_or_malformed_output_falls_back(self):
        original = "What happened afterward?"
        for response in [LLMRequestError("provider unavailable"), "first line\nsecond line"]:
            model = SequencedModel([response])
            query = QueryContextualizer(model, Settings()).contextualize(
                original,
                [{"role": "user", "content": "Earlier question"}],
            )
            self.assertEqual(query, "Earlier question What happened afterward?")

    def test_rewrite_failure_without_prior_user_question_keeps_original(self):
        model = SequencedModel([LLMRequestError("provider unavailable")])
        query = QueryContextualizer(model, Settings()).contextualize(
            "Και μετά;",
            [{"role": "assistant", "content": "Δεν υπάρχουν επαρκή στοιχεία."}],
        )
        self.assertEqual(query, "Και μετά;")

    def test_pipeline_retrieves_with_rewrite_but_answers_original_question(self):
        retriever = RecordingRetriever()
        model = SequencedModel(
            [
                "What happened after the reported warnings before the Tempi accident?",
                "The sources describe subsequent events [S1].",
            ]
        )
        result = GroundedRAG(
            Settings(), retriever=retriever, chat_model=model
        ).answer(
            "What happened afterward?",
            history=[{"role": "user", "content": "What warnings were reported before the Tempi accident?"}],
        )
        self.assertEqual(result.status, AnswerStatus.ANSWERED)
        self.assertEqual(
            retriever.queries[0][0],
            "What happened after the reported warnings before the Tempi accident?",
        )
        self.assertEqual(result.retrieval_query, retriever.queries[0][0])
        self.assertIn("QUESTION:\nWhat happened afterward?", model.messages[1][1]["content"])


if __name__ == "__main__":
    unittest.main()
