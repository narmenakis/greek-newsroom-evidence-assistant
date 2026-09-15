import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from journalism_rag.config import Settings
from journalism_rag.llm import LLMToolValidationError, ToolDefinition
from journalism_rag.mcp_web import TAVILY_EXTRACT_TOOL, TAVILY_SEARCH_TOOL
from journalism_rag.providers.llm import OllamaChatModel
from journalism_rag.research_brief import (
    BRIEF_SECTIONS,
    MAX_WEB_EVIDENCE_CHARACTERS,
    MAX_WEB_EVIDENCE_URLS,
    MAX_MCP_CALLS,
    ResearchBriefAgent,
    ResearchBriefBudgetError,
)
from journalism_rag.web_sources import WebEvidence


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


class FakeRAG:
    def __init__(self, chat_model):
        self.settings = Settings(
            llm_provider="ollama",
            llm_model="journalism-rag-qwen3.5:9b",
            llm_base_url="http://localhost:11434/v1",
        )
        self.chat_model = chat_model

    def retrieve_context(self, query, **kwargs):
        del kwargs
        return [], "Archive context", (), query.strip()


class StrictMCP:
    """Deterministic Tavily double with the production-shaped schemas."""

    def __init__(self):
        self.calls = []

    def discover_tools(self):
        result_schema = {
            "type": "object",
            "properties": {"results": {"type": "array"}},
            "required": ["results"],
            "additionalProperties": False,
        }
        return (
            ToolDefinition(
                name=TAVILY_SEARCH_TOOL,
                schema_version="tavily-mcp-1",
                description="Search current web reporting",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "minLength": 1},
                        "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                output_schema=result_schema,
            ),
            ToolDefinition(
                name=TAVILY_EXTRACT_TOOL,
                schema_version="tavily-mcp-1",
                description="Extract text from web pages",
                input_schema={
                    "type": "object",
                    "properties": {
                        "urls": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 1},
                            "minItems": 1,
                        }
                    },
                    "required": ["urls"],
                    "additionalProperties": False,
                },
                output_schema=result_schema,
            ),
        )

    def search(self, query, *, max_results=5):
        self.calls.append((TAVILY_SEARCH_TOOL, query, max_results))
        return (
            WebEvidence(
                title="Current report",
                url="https://example.com/current",
                excerpt="Current web evidence",
                retrieved_at=datetime(2026, 9, 10, 12, tzinfo=timezone.utc),
            ),
        )

    def extract(self, urls):
        self.calls.append((TAVILY_EXTRACT_TOOL, tuple(urls)))
        return ()


def tool_call_body(call_id, name, arguments):
    return FakeResponse(
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": arguments,
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
    )


def final_body():
    brief = "\n\n".join(
        f"## {section}\nFinding supported by [W1]." for section in BRIEF_SECTIONS
    )
    return FakeResponse(
        {"choices": [{"message": {"content": brief}, "finish_reason": "stop"}]}
    )


class Phase8ResearchAgentTests(unittest.TestCase):
    def test_ollama_runs_bounded_research_loop_with_tavily_evidence(self):
        session = QueueSession(
            tool_call_body(
                "local-extract",
                TAVILY_EXTRACT_TOOL,
                '{"urls":["https://example.com/current"]}',
            ),
            final_body(),
        )
        model = OllamaChatModel(
            Settings(
                llm_provider="ollama",
                llm_model="journalism-rag-qwen3.5:9b",
                llm_base_url="http://localhost:11434/v1",
            ),
            session=session,
        )
        mcp = StrictMCP()

        run = ResearchBriefAgent(FakeRAG(model), mcp_client=mcp).run("latest developments")

        self.assertEqual(run.generation.provider, "ollama")
        self.assertEqual(run.mcp_calls, 2)  # required search plus one model-selected call
        self.assertEqual(len(run.tool_exchanges), 2)
        self.assertEqual(len(mcp.calls), 2)
        self.assertEqual(run.citation_validation.cited_source_ids, ("W1",))
        self.assertTrue(all(f"## {section}" in run.brief for section in BRIEF_SECTIONS))
        follow_up = session.calls[1][1]["json"]["messages"]
        self.assertEqual(follow_up[-1]["role"], "tool")
        self.assertEqual(follow_up[-1]["tool_call_id"], "local-extract")

    def test_ollama_tool_arguments_still_fail_schema_validation(self):
        session = QueueSession(
            tool_call_body(
                "bad-local-call",
                TAVILY_EXTRACT_TOOL,
                '{"urls":"https://example.com/current"}',
            )
        )
        model = OllamaChatModel(
            Settings(
                llm_provider="ollama",
                llm_model="journalism-rag-qwen3.5:9b",
                llm_base_url="http://localhost:11434/v1",
            ),
            session=session,
        )
        mcp = StrictMCP()

        with self.assertRaises(LLMToolValidationError):
            ResearchBriefAgent(FakeRAG(model), mcp_client=mcp).run("latest developments")

        self.assertEqual(len(mcp.calls), 1)  # only the mandatory search ran

    def test_ollama_loop_stops_after_four_mcp_calls(self):
        responses = []
        for index in range(1, MAX_MCP_CALLS + 1):
            if index % 2:
                responses.append(
                    tool_call_body(
                        f"local-search-{index}",
                        TAVILY_SEARCH_TOOL,
                        '{"query":"latest","max_results":2}',
                    )
                )
            else:
                responses.append(
                    tool_call_body(
                        f"local-extract-{index}",
                        TAVILY_EXTRACT_TOOL,
                        '{"urls":["https://example.com/current"]}',
                    )
                )
        session = QueueSession(*responses)
        model = OllamaChatModel(
            Settings(
                llm_provider="ollama",
                llm_model="journalism-rag-qwen3.5:9b",
                llm_base_url="http://localhost:11434/v1",
            ),
            session=session,
        )
        mcp = StrictMCP()

        with self.assertRaises(ResearchBriefBudgetError):
            ResearchBriefAgent(
                FakeRAG(model), mcp_client=mcp, max_mcp_calls=MAX_MCP_CALLS
            ).run("keep searching")

        self.assertEqual(len(mcp.calls), MAX_MCP_CALLS)
        self.assertEqual(len(session.calls), MAX_MCP_CALLS)

    def test_research_rejects_oversized_aggregate_web_evidence(self):
        class OversizedMCP(StrictMCP):
            def search(self, query, *, max_results=5):
                del query, max_results
                return tuple(
                    WebEvidence(
                        title=f"Result {index}",
                        url=f"https://example.com/result-{index}",
                        excerpt="x" * (MAX_WEB_EVIDENCE_CHARACTERS // 3 + 1000),
                        retrieved_at=datetime(2026, 9, 10, 12, tzinfo=timezone.utc),
                    )
                    for index in range(3)
                )

        model = OllamaChatModel(
            Settings(
                llm_provider="ollama",
                llm_model="journalism-rag-qwen3.5:9b",
                llm_base_url="http://localhost:11434/v1",
            ),
            session=QueueSession(),
        )

        with self.assertRaisesRegex(ResearchBriefBudgetError, "character"):
            ResearchBriefAgent(FakeRAG(model), mcp_client=OversizedMCP()).run("large topic")

    def test_research_appends_incremental_web_context_with_stable_ids(self):
        class IncrementalMCP(StrictMCP):
            def __init__(self):
                super().__init__()
                self.extract_count = 0

            def extract(self, urls):
                self.calls.append((TAVILY_EXTRACT_TOOL, tuple(urls)))
                self.extract_count += 1
                index = self.extract_count + 1
                return (
                    WebEvidence(
                        title=f"Current report {index}",
                        url=f"https://example.com/current-{index}",
                        excerpt=f"Web evidence {index}",
                        retrieved_at=datetime(2026, 9, 10, 12, tzinfo=timezone.utc),
                    ),
                )

        session = QueueSession(
            tool_call_body(
                "extract-1",
                TAVILY_EXTRACT_TOOL,
                '{"urls":["https://example.com/current"]}',
            ),
            tool_call_body(
                "extract-2",
                TAVILY_EXTRACT_TOOL,
                '{"urls":["https://example.com/current-2"]}',
            ),
            final_body(),
        )
        mcp = IncrementalMCP()
        model = OllamaChatModel(
            Settings(
                llm_provider="ollama",
                llm_model="journalism-rag-qwen3.5:9b",
                llm_base_url="http://localhost:11434/v1",
            ),
            session=session,
        )

        run = ResearchBriefAgent(FakeRAG(model), mcp_client=mcp).run("incremental topic")

        self.assertEqual(run.mcp_calls, 3)
        second_request = session.calls[1][1]["json"]["messages"]
        third_request = session.calls[2][1]["json"]["messages"]
        second_contexts = [
            message["content"]
            for message in second_request
            if "CURRENT WEB EVIDENCE:" in (message.get("content") or "")
        ]
        third_contexts = [
            message["content"]
            for message in third_request
            if "CURRENT WEB EVIDENCE:" in (message.get("content") or "")
        ]
        self.assertEqual(len(second_contexts), 2)
        self.assertEqual(len(third_contexts), 3)
        self.assertIn("[W2]", second_contexts[-1])
        self.assertNotIn("[W1]", second_contexts[-1])
        self.assertIn("[W3]", third_contexts[-1])
        self.assertNotIn("[W1]", third_contexts[-1])

    def test_research_rejects_too_many_extract_targets_before_mcp_call(self):
        urls = [
            f"https://example.com/target-{index}"
            for index in range(MAX_WEB_EVIDENCE_URLS + 1)
        ]
        session = QueueSession(
            tool_call_body(
                "too-many-urls",
                TAVILY_EXTRACT_TOOL,
                json.dumps({"urls": urls}),
            )
        )
        mcp = StrictMCP()
        model = OllamaChatModel(
            Settings(
                llm_provider="ollama",
                llm_model="journalism-rag-qwen3.5:9b",
                llm_base_url="http://localhost:11434/v1",
            ),
            session=session,
        )

        with self.assertRaisesRegex(ResearchBriefBudgetError, "URL web extraction"):
            ResearchBriefAgent(FakeRAG(model), mcp_client=mcp).run("many targets")

        # The mandatory search ran, but the oversized extract was rejected
        # before it reached the MCP client.
        self.assertEqual(len(mcp.calls), 1)


if __name__ == "__main__":
    unittest.main()
