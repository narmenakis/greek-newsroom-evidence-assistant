import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from journalism_rag.llm import (
    ChatModel,
    GenerationResult,
    GenerationStreamEvent,
    LLMStreamError,
    LLMStreamingUnsupportedError,
)


class FakeChatModel(ChatModel):
    provider = "fake"
    model = "fake-model"

    def generate(self, messages, options=None, *, tools=None, tool_history=None):
        del messages, options, tools, tool_history
        return GenerationResult(
            text="complete",
            provider=self.provider,
            model=self.model,
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            latency_seconds=0.01,
        )


class Phase7StreamingContractTests(unittest.TestCase):
    def test_events_have_explicit_normalized_shapes(self):
        result = FakeChatModel().generate([{"role": "user", "content": "hi"}])
        self.assertEqual(GenerationStreamEvent.delta("hel").event_type, "delta")
        self.assertEqual(
            GenerationStreamEvent.completed(result).result.text,
            "complete",
        )
        failure = LLMStreamError("connection ended")
        self.assertIs(GenerationStreamEvent.failed(failure).error, failure)

    def test_invalid_event_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            GenerationStreamEvent(event_type="delta")
        with self.assertRaises(ValueError):
            GenerationStreamEvent(event_type="completed")
        with self.assertRaises(ValueError):
            GenerationStreamEvent(event_type="error")

    def test_unimplemented_provider_fails_explicitly(self):
        model = FakeChatModel()
        self.assertFalse(model.supports_streaming)
        with self.assertRaises(LLMStreamingUnsupportedError):
            list(model.stream([{"role": "user", "content": "hi"}]))


if __name__ == "__main__":
    unittest.main()
