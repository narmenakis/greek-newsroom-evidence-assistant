import sys
import unittest
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from journalism_rag.config import Settings
from journalism_rag.llm import (
    GenerationOptions,
    LLMStreamProtocolError,
    LLMStreamingUnsupportedError,
    ToolDefinition,
)
from journalism_rag.providers.llm import DeepSeekChatModel, OllamaChatModel


class FakeStreamResponse:
    def __init__(self, lines, *, status_code=200, text=""):
        self.lines = lines
        self.status_code = status_code
        self.text = text
        self.closed = False

    def iter_lines(self, *, decode_unicode):
        del decode_unicode
        return iter(self.lines)

    def close(self):
        self.closed = True


class FakeStreamSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.response


def stream_lines():
    return [
        'data: {"choices":[{"delta":{"content":"Γεια "},"finish_reason":null}]}',
        'data: {"choices":[{"delta":{"content":"κόσμε"},"finish_reason":null}]}',
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6}}',
        "data: [DONE]",
    ]


def ollama_stream_lines():
    """Representative Ollama OpenAI stream, including its usage-only event."""

    return [
        'data: {"choices":[{"delta":{"role":"assistant","content":"","reasoning":"Thinking"},"finish_reason":null}]}',
        'data: {"choices":[{"delta":{"content":"Γεια"},"finish_reason":null}]}',
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
        'data: {"choices":[],"usage":{"prompt_tokens":4,"completion_tokens":1,"total_tokens":5}}',
        "data: [DONE]",
    ]


def latin1_inferred_sse_response():
    """Build the response shape that previously split Greek UTF-8 mid-event."""

    lines = [
        'data: {"choices":[{"delta":{"content":"του"},"finish_reason":"stop"}]}',
        'data: {"choices":[],"usage":{"prompt_tokens":4,"completion_tokens":1,"total_tokens":5}}',
        "data: [DONE]",
    ]
    response = requests.Response()
    response.status_code = 200
    response.encoding = "ISO-8859-1"
    response._content = ("\n\n".join(lines) + "\n\n").encode("utf-8")
    response._content_consumed = True
    return response


class Phase7StreamingAdapterTests(unittest.TestCase):
    def test_deepseek_and_ollama_map_sse_to_normalized_events(self):
        for model_class, settings in (
            (DeepSeekChatModel, Settings(deepseek_api_key="secret")),
            (
                OllamaChatModel,
                Settings(
                    llm_provider="ollama",
                    llm_model="journalism-rag-qwen3.5:9b",
                    llm_base_url="http://localhost:11434/v1",
                ),
            ),
        ):
            with self.subTest(provider=model_class.provider):
                response = FakeStreamResponse(stream_lines())
                session = FakeStreamSession(response)
                model = model_class(settings, session=session)
                events = list(
                    model.stream(
                        [{"role": "user", "content": "Χαιρετισμός"}],
                        GenerationOptions(max_tokens=16),
                    )
                )
                self.assertEqual([event.event_type for event in events], ["delta", "delta", "completed"])
                self.assertEqual(events[-1].result.text, "Γεια κόσμε")
                self.assertEqual(events[-1].result.total_tokens, 6)
                self.assertTrue(response.closed)
                payload = session.calls[0][1]["json"]
                self.assertTrue(payload["stream"])
                self.assertEqual(payload["max_tokens"], 16)
                self.assertTrue(session.calls[0][1]["stream"])

    def test_malformed_or_incomplete_sse_emits_protocol_error(self):
        for lines in (
            ["data: {not-json}"],
            ['data: {"choices":[{"delta":{"content":"partial"}}]}'],
        ):
            with self.subTest(lines=lines):
                response = FakeStreamResponse(lines)
                model = OllamaChatModel(
                    Settings(
                        llm_provider="ollama",
                        llm_model="journalism-rag-qwen3.5:9b",
                        llm_base_url="http://localhost:11434/v1",
                    ),
                    session=FakeStreamSession(response),
                )
                events = list(model.stream([{"role": "user", "content": "hi"}]))
                self.assertEqual(events[-1].event_type, "error")
                self.assertIsInstance(events[-1].error, LLMStreamProtocolError)

    def test_ollama_usage_only_event_is_not_treated_as_malformed(self):
        model = OllamaChatModel(
            Settings(
                llm_provider="ollama",
                llm_model="journalism-rag-qwen3.5:9b",
                llm_base_url="http://localhost:11434/v1",
            ),
            session=FakeStreamSession(FakeStreamResponse(ollama_stream_lines())),
        )
        events = list(model.stream([{"role": "user", "content": "hi"}]))
        self.assertEqual(
            [event.event_type for event in events],
            ["delta", "completed"],
        )
        self.assertEqual(events[-1].result.text, "Γεια")
        self.assertEqual(events[-1].result.total_tokens, 5)

    def test_greek_utf8_is_decoded_after_sse_lines_are_split(self):
        model = OllamaChatModel(
            Settings(
                llm_provider="ollama",
                llm_model="journalism-rag-qwen3.5:9b",
                llm_base_url="http://localhost:11434/v1",
            ),
            session=FakeStreamSession(latin1_inferred_sse_response()),
        )
        events = list(model.stream([{"role": "user", "content": "hi"}]))
        self.assertEqual(
            [event.event_type for event in events],
            ["delta", "completed"],
        )
        self.assertEqual(events[0].text, "του")
        self.assertEqual(events[-1].result.text, "του")

    def test_deepseek_stream_uses_configured_thinking_default(self):
        response = FakeStreamResponse(stream_lines())
        session = FakeStreamSession(response)
        model = DeepSeekChatModel(
            Settings(deepseek_api_key="secret", deepseek_thinking="enabled"),
            session=session,
        )
        list(model.stream([{"role": "user", "content": "hi"}]))
        self.assertEqual(session.calls[0][1]["json"]["thinking"], {"type": "enabled"})

    def test_streaming_tool_calls_are_rejected_explicitly(self):
        model = OllamaChatModel(
            Settings(
                llm_provider="ollama",
                llm_model="journalism-rag-qwen3.5:9b",
                llm_base_url="http://localhost:11434/v1",
            ),
            session=FakeStreamSession(FakeStreamResponse([])),
        )
        with self.assertRaises(LLMStreamingUnsupportedError):
            tool = ToolDefinition(
                name="example",
                schema_version="1.0",
                description="Example tool",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
            )
            list(model.stream([{"role": "user", "content": "hi"}], tools=[tool]))


if __name__ == "__main__":
    unittest.main()
