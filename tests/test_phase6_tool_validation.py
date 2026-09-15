import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from journalism_rag.llm import (
    LLMToolValidationError,
    ToolCall,
    ToolDefinition,
    ToolExchange,
    ToolResult,
)
from journalism_rag.config import Settings
from journalism_rag.providers.llm import DeepSeekChatModel
from journalism_rag.tool_validation import (
    validate_tool_arguments,
    validate_tool_definition,
    validate_tool_result,
)


def search_definition() -> ToolDefinition:
    return ToolDefinition(
        name="search_corpus",
        schema_version="1.0",
        description="Search the permanent journalism corpus",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "chunk_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "status": {"type": "string", "enum": ["ok", "partial", "error"]},
            },
            "required": ["chunk_ids", "status"],
            "additionalProperties": False,
        },
    )


class FakeResponse:
    status_code = 200
    text = ""

    def __init__(self, body):
        self.body = body

    def json(self):
        return self.body


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.response


class Phase6ToolValidationTests(unittest.TestCase):
    def test_valid_definition_arguments_and_result_pass(self):
        definition = search_definition()
        call = ToolCall(
            call_id="call_1",
            name="search_corpus",
            arguments={"query": "Τέμπη", "limit": 5},
        )
        result = ToolResult(
            call_id="call_1",
            name="search_corpus",
            content={"chunk_ids": ["chunk_1"], "status": "ok"},
        )

        validate_tool_definition(definition)
        validate_tool_arguments(definition, call)
        validate_tool_result(definition, result)

    def test_arguments_reject_wrong_types_bounds_and_unknown_fields(self):
        definition = search_definition()
        invalid_arguments = (
            {"query": 123},
            {"query": "Τέμπη", "limit": 100},
            {"query": "Τέμπη", "delete_database": True},
        )

        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises(LLMToolValidationError):
                    validate_tool_arguments(
                        definition,
                        ToolCall(
                            call_id="call_bad",
                            name="search_corpus",
                            arguments=arguments,
                        ),
                    )

    def test_result_rejects_malformed_content(self):
        with self.assertRaisesRegex(LLMToolValidationError, "rejected result"):
            validate_tool_result(
                search_definition(),
                ToolResult(
                    call_id="call_bad",
                    name="search_corpus",
                    content={"chunk_ids": "not-an-array", "status": "ok"},
                ),
            )

    def test_invalid_schema_and_empty_version_fail_closed(self):
        invalid_schema = ToolDefinition(
            name="search_corpus",
            schema_version="1.0",
            description="Search",
            input_schema={"type": "not-a-json-schema-type"},
            output_schema={"type": "object"},
        )
        empty_version = ToolDefinition(
            name="search_corpus",
            schema_version="",
            description="Search",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )

        with self.assertRaisesRegex(LLMToolValidationError, "invalid input schema"):
            validate_tool_definition(invalid_schema)
        with self.assertRaisesRegex(LLMToolValidationError, "version cannot be empty"):
            validate_tool_definition(empty_version)

    def test_provider_rejects_invalid_model_arguments_before_returning_them(self):
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
                                            "arguments": '{"query":123}',
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

        with self.assertRaisesRegex(LLMToolValidationError, "rejected arguments"):
            DeepSeekChatModel(
                Settings(deepseek_api_key="fake-secret"), session=session
            ).generate(
                [{"role": "user", "content": "Search"}],
                tools=[search_definition()],
            )

    def test_provider_rejects_invalid_result_before_sending_model_context(self):
        session = FakeSession(FakeResponse({}))
        invalid_result = ToolResult(
            call_id="call_1",
            name="search_corpus",
            content={"chunk_ids": "not-an-array", "status": "ok"},
        )
        history = ToolExchange(
            calls=(
                ToolCall(
                    call_id="call_1",
                    name="search_corpus",
                    arguments={"query": "Τέμπη"},
                ),
            ),
            results=(invalid_result,),
        )

        with self.assertRaisesRegex(LLMToolValidationError, "rejected result"):
            DeepSeekChatModel(
                Settings(deepseek_api_key="fake-secret"), session=session
            ).generate(
                [{"role": "user", "content": "Search"}],
                tools=[search_definition()],
                tool_history=[history],
            )

        self.assertEqual(session.calls, [])


if __name__ == "__main__":
    unittest.main()
