import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from journalism_rag.config import Settings
from journalism_rag.llm import GenerationOptions, ToolCall, ToolDefinition, ToolExchange, ToolResult
from journalism_rag.providers.llm import OllamaChatModel


class FakeResponse:
    status_code = 200
    text = ""

    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body


class QueueSession:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.responses.pop(0)


def temperature_tool():
    return ToolDefinition(
        name="get_temperature",
        schema_version="1.0",
        description="Get the current temperature for a city",
        input_schema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {"temperature_c": {"type": "number"}},
            "required": ["temperature_c"],
            "additionalProperties": False,
        },
    )


class Phase8OllamaToolTests(unittest.TestCase):
    def make_model(self, session):
        return OllamaChatModel(
            Settings(
                llm_provider="ollama",
                llm_model="journalism-rag-qwen3.5:9b",
                llm_base_url="http://localhost:11434/v1",
            ),
            session=session,
        )

    def test_tool_definition_is_sent_and_tool_call_is_parsed(self):
        session = QueueSession(
            FakeResponse(
                {
                    "model": "journalism-rag-qwen3.5:9b",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_local",
                                        "type": "function",
                                        "function": {
                                            "name": "get_temperature",
                                            "arguments": '{"city":"Athens"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
                }
            )
        )
        tool = temperature_tool()

        result = self.make_model(session).generate(
            [{"role": "user", "content": "What is the temperature in Athens?"}],
            GenerationOptions(max_tokens=64),
            tools=[tool],
        )

        self.assertEqual(
            result.tool_calls,
            (ToolCall(call_id="call_local", name="get_temperature", arguments={"city": "Athens"}),),
        )
        payload = session.calls[0][1]["json"]
        self.assertEqual(payload["tools"][0]["function"]["name"], "get_temperature")
        self.assertEqual(payload["tools"][0]["function"]["parameters"], dict(tool.input_schema))

    def test_tool_result_history_is_sent_and_final_answer_is_parsed(self):
        session = QueueSession(
            FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_local",
                                        "type": "function",
                                        "function": {
                                            "name": "get_temperature",
                                            "arguments": '{"city":"Athens"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                }
            ),
            FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "The current temperature in Athens is 25°C."
                            },
                            "finish_reason": "stop",
                        }
                    ]
                }
            ),
        )
        tool = temperature_tool()
        model = self.make_model(session)
        messages = [{"role": "user", "content": "What is the temperature in Athens?"}]
        first = model.generate(messages, tools=[tool])
        exchange = ToolExchange(
            calls=first.tool_calls,
            results=(
                ToolResult(
                    call_id="call_local",
                    name="get_temperature",
                    content={"temperature_c": 25},
                ),
            ),
        )

        final = model.generate(messages, tools=[tool], tool_history=[exchange])

        self.assertEqual(final.text, "The current temperature in Athens is 25°C.")
        follow_up_messages = session.calls[1][1]["json"]["messages"]
        self.assertEqual(follow_up_messages[-2]["role"], "assistant")
        self.assertEqual(follow_up_messages[-2]["tool_calls"][0]["id"], "call_local")
        self.assertEqual(follow_up_messages[-1]["role"], "tool")
        self.assertEqual(follow_up_messages[-1]["tool_call_id"], "call_local")
        self.assertIn('"temperature_c": 25', follow_up_messages[-1]["content"])


if __name__ == "__main__":
    unittest.main()
