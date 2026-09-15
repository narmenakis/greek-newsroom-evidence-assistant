import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from journalism_rag.config import Settings
from journalism_rag.llm import (
    ChatModel,
    GenerationResult,
    LLMRequestError,
    LLMToolError,
    LLMToolResponseError,
    LLMToolUnsupportedError,
    ToolCall,
    ToolDefinition,
    ToolExchange,
    ToolResult,
)
from journalism_rag.providers.llm import DeepSeekChatModel, OllamaChatModel


class FakeResponse:
    def __init__(self, body):
        self.status_code = 200
        self._body = body
        self.text = ""

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.response


class FakeToolAwareModel(ChatModel):
    provider = "fake"
    model = "fake-tool-model"

    def __init__(self):
        self.tools = None
        self.tool_history = None

    def generate(self, messages, options=None, *, tools=None, tool_history=None):
        self.tools = tools
        self.tool_history = tool_history
        if not tools:
            return GenerationResult(
                text="Direct answer",
                provider=self.provider,
                model=self.model,
                finish_reason="stop",
                prompt_tokens=6,
                completion_tokens=2,
                total_tokens=8,
                latency_seconds=0.01,
            )
        return GenerationResult(
            text="",
            provider=self.provider,
            model=self.model,
            finish_reason="tool_calls",
            prompt_tokens=10,
            completion_tokens=4,
            total_tokens=14,
            latency_seconds=0.01,
            tool_calls=(
                ToolCall(
                    call_id="call_123",
                    name="search_corpus",
                    arguments={"query": "τηλεδιοίκηση στα Τέμπη"},
                ),
            ),
        )


class Phase6ToolContractTests(unittest.TestCase):
    def test_contract_preserves_structured_definitions_calls_and_results(self):
        definition = ToolDefinition(
            name="search_corpus",
            schema_version="1.0",
            description="Search the permanent journalism corpus",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            output_schema={"type": "object"},
        )
        previous_exchange = ToolExchange(
            calls=(
                ToolCall(
                    call_id="call_previous",
                    name="search_corpus",
                    arguments={"query": "Τέμπη"},
                ),
            ),
            results=(
                ToolResult(
                    call_id="call_previous",
                    name="search_corpus",
                    content={"chunk_ids": ["chunk_1"], "status": "ok"},
                ),
            ),
        )
        model = FakeToolAwareModel()

        result = model.generate(
            [{"role": "user", "content": "Research railway safety"}],
            tools=[definition],
            tool_history=[previous_exchange],
        )

        self.assertEqual(model.tools, [definition])
        self.assertEqual(model.tool_history, [previous_exchange])
        self.assertEqual(result.text, "")
        self.assertEqual(result.tool_calls[0].call_id, "call_123")
        self.assertEqual(result.tool_calls[0].name, "search_corpus")
        self.assertEqual(
            result.tool_calls[0].arguments,
            {"query": "τηλεδιοίκηση στα Τέμπη"},
        )

    def test_tool_errors_remain_normalized_llm_request_errors(self):
        self.assertTrue(issubclass(LLMToolError, LLMRequestError))
        self.assertTrue(issubclass(LLMToolUnsupportedError, LLMToolError))
        self.assertTrue(issubclass(LLMToolResponseError, LLMToolError))

    def test_local_provider_advertises_verified_non_streaming_tool_support(self):
        model = OllamaChatModel(
            Settings(
                llm_provider="ollama",
                llm_model="local-model",
                llm_base_url="http://localhost:11434/v1",
            )
        )
        self.assertTrue(model.supports_tool_calling)
        self.assertTrue(model.supports_streaming)

    def test_deepseek_maps_tool_definition_and_tool_result_to_request(self):
        session = FakeSession(
            FakeResponse(
                {
                    "choices": [
                        {"message": {"content": "Final answer"}, "finish_reason": "stop"}
                    ]
                }
            )
        )
        definition = ToolDefinition(
            name="search_corpus",
            schema_version="1.0",
            description="Search the permanent journalism corpus",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            output_schema={"type": "object"},
        )
        tool_result = ToolResult(
            call_id="call_123",
            name="search_corpus",
            content={"chunk_ids": ["chunk_1"], "status": "ok"},
        )
        exchange = ToolExchange(
            calls=(
                ToolCall(
                    call_id="call_123",
                    name="search_corpus",
                    arguments={"query": "Τέμπη"},
                ),
            ),
            results=(tool_result,),
        )

        DeepSeekChatModel(
            Settings(deepseek_api_key="fake-secret"), session=session
        ).generate(
            [{"role": "user", "content": "Research railway safety"}],
            tools=[definition],
            tool_history=[exchange],
        )

        payload = session.calls[0][1]["json"]
        self.assertEqual(
            payload["tools"],
            [
                {
                    "type": "function",
                    "function": {
                        "name": "search_corpus",
                        "description": "Search the permanent journalism corpus",
                        "parameters": definition.input_schema,
                    },
                }
            ],
        )
        self.assertEqual(
            payload["messages"][-2],
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "search_corpus",
                            "arguments": '{"query": "Τέμπη"}',
                        },
                    }
                ],
            },
        )
        self.assertEqual(
            payload["messages"][-1],
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "content": (
                    '{"is_error": false, "result": '
                    '{"chunk_ids": ["chunk_1"], "status": "ok"}}'
                ),
            },
        )

    def test_deepseek_parses_tool_call_response(self):
        session = FakeSession(
            FakeResponse(
                {
                    "model": "deepseek-flash",
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_123",
                                        "type": "function",
                                        "function": {
                                            "name": "search_corpus",
                                            "arguments": '{"query":"Τέμπη"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 5,
                        "total_tokens": 17,
                    },
                }
            )
        )
        definition = ToolDefinition(
            name="search_corpus",
            schema_version="1.0",
            description="Search the corpus",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )

        result = DeepSeekChatModel(
            Settings(deepseek_api_key="fake-secret"), session=session
        ).generate(
            [{"role": "user", "content": "Search"}],
            tools=[definition],
        )

        self.assertEqual(result.text, "")
        self.assertEqual(result.finish_reason, "tool_calls")
        self.assertEqual(
            result.tool_calls,
            (
                ToolCall(
                    call_id="call_123",
                    name="search_corpus",
                    arguments={"query": "Τέμπη"},
                ),
            ),
        )

    def test_deepseek_rejects_malformed_tool_arguments(self):
        session = FakeSession(
            FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_bad",
                                        "type": "function",
                                        "function": {
                                            "name": "search_corpus",
                                            "arguments": "not-json",
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                }
            )
        )

        with self.assertRaisesRegex(LLMToolResponseError, "not valid JSON"):
            DeepSeekChatModel(
                Settings(deepseek_api_key="fake-secret"), session=session
            ).generate(
                [{"role": "user", "content": "Search"}],
                tools=[
                    ToolDefinition(
                        name="search_corpus",
                        schema_version="1.0",
                        description="Search the corpus",
                        input_schema={"type": "object"},
                        output_schema={"type": "object"},
                    )
                ],
            )

    def test_existing_direct_generation_contract_needs_no_tools(self):
        model = FakeToolAwareModel()

        result = model.generate([{"role": "user", "content": "Answer directly"}])

        self.assertIsNone(model.tools)
        self.assertIsNone(model.tool_history)
        self.assertEqual(result.text, "Direct answer")
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.tool_calls, ())


if __name__ == "__main__":
    unittest.main()
