import unittest
from datetime import datetime, timezone

from langchain_core.documents import Document

from journalism_rag.llm import GenerationResult, ToolCall, ToolDefinition
from journalism_rag.mcp_web import TAVILY_EXTRACT_TOOL, TAVILY_SEARCH_TOOL
from journalism_rag.pipeline import GroundedRAG
from journalism_rag.research_brief import (
    BRIEF_SECTIONS,
    MAX_MCP_CALLS,
    ResearchBriefAgent,
    ResearchBriefBudgetError,
    ResearchBriefError,
    ResearchBriefValidationError,
)
from journalism_rag.config import Settings
from journalism_rag.retrieval_types import RetrievedChunk
from journalism_rag.web_sources import WebEvidence


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def retrieve(self, query, *, limit=5, where=None):
        self.calls.append((query, limit, where))
        return [
            RetrievedChunk(
                document=Document(
                    page_content="Archive context",
                    metadata={"article_id": "a1", "chunk_id": "c1", "title": "Archive"},
                ),
                score=1.0,
                rank=1,
            )
        ]


def definition(name):
    return ToolDefinition(
        name=name,
        schema_version="1",
        description=name,
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )


class FakeMCP:
    def __init__(self, *, with_evidence=True):
        self.calls = []
        self.with_evidence = with_evidence

    def discover_tools(self):
        return (definition(TAVILY_SEARCH_TOOL), definition(TAVILY_EXTRACT_TOOL))

    def search(self, query, *, max_results=5):
        self.calls.append((TAVILY_SEARCH_TOOL, query, max_results))
        if not self.with_evidence:
            return ()
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


class FakeModel:
    provider = "fake"
    model = "fake-model"

    def __init__(self, *, always_call=False, cite_web=True):
        self.calls = []
        self.always_call = always_call
        self.cite_web = cite_web

    def generate(self, messages, options=None, *, tools=None, tool_history=None):
        self.calls.append({"messages": messages, "tools": tools, "history": tool_history})
        if self.always_call:
            name = TAVILY_SEARCH_TOOL if len(self.calls) % 2 else TAVILY_EXTRACT_TOOL
            arguments = {"query": "latest", "max_results": 2} if name == TAVILY_SEARCH_TOOL else {"urls": ["https://example.com"]}
            return GenerationResult(
                text="",
                provider=self.provider,
                model=self.model,
                finish_reason="tool_calls",
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                latency_seconds=0,
                tool_calls=(ToolCall(call_id=f"call-{len(self.calls)}", name=name, arguments=arguments),),
            )
        citation = "[W1]" if self.cite_web else "[S1]"
        brief = "\n\n".join(f"## {section}\nFinding {citation}." for section in BRIEF_SECTIONS)
        return GenerationResult(
            text=brief,
            provider=self.provider,
            model=self.model,
            finish_reason="stop",
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            latency_seconds=0,
        )


class Phase6ResearchBriefTests(unittest.TestCase):
    def test_retrieves_local_context_once_and_runs_tool_then_final_generation(self):
        retriever = FakeRetriever()
        model = FakeModel()
        mcp = FakeMCP()
        rag = GroundedRAG(Settings(), retriever=retriever, chat_model=model)

        run = ResearchBriefAgent(rag, mcp_client=mcp).run("latest developments")

        self.assertEqual(len(retriever.calls), 1)
        self.assertEqual(run.mcp_calls, 1)
        self.assertEqual(len(run.tool_exchanges), 1)
        self.assertEqual(run.brief.splitlines()[0], "## Τρέχουσες εξελίξεις")
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(len(model.calls[0]["history"]), 1)
        self.assertEqual(
            mcp.calls[0],
            (TAVILY_SEARCH_TOOL, "latest developments", 5),
        )

    def test_web_evidence_gets_w_citation_and_enters_final_context(self):
        model = FakeModel(cite_web=True)
        mcp = FakeMCP(with_evidence=True)
        rag = GroundedRAG(Settings(), retriever=FakeRetriever(), chat_model=model)

        run = ResearchBriefAgent(rag, mcp_client=mcp).run("latest developments")

        self.assertEqual([source.source_id for source in run.sources], ["S1", "W1"])
        self.assertEqual(run.citation_validation.cited_source_ids, ("W1",))
        self.assertIn("[W1]", model.calls[0]["messages"][-1]["content"])

    def test_fails_closed_when_required_search_returns_no_evidence(self):
        model = FakeModel()
        rag = GroundedRAG(Settings(), retriever=FakeRetriever(), chat_model=model)

        with self.assertRaisesRegex(ResearchBriefError, "no usable evidence"):
            ResearchBriefAgent(
                rag,
                mcp_client=FakeMCP(with_evidence=False),
            ).run("latest developments")

        self.assertEqual(model.calls, [])

    def test_fails_closed_when_final_brief_does_not_cite_web_evidence(self):
        model = FakeModel(cite_web=False)
        rag = GroundedRAG(Settings(), retriever=FakeRetriever(), chat_model=model)

        with self.assertRaisesRegex(
            ResearchBriefValidationError,
            "no web source was cited",
        ):
            ResearchBriefAgent(rag, mcp_client=FakeMCP()).run(
                "latest developments"
            )

    def test_stops_by_failing_closed_when_model_ignores_call_limit(self):
        model = FakeModel(always_call=True)
        rag = GroundedRAG(Settings(), retriever=FakeRetriever(), chat_model=model)

        with self.assertRaises(ResearchBriefBudgetError):
            ResearchBriefAgent(
                rag,
                mcp_client=FakeMCP(),
                max_mcp_calls=MAX_MCP_CALLS,
            ).run("keep searching")


if __name__ == "__main__":
    unittest.main()
