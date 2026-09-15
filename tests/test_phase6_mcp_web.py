import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from mcp.types import TextContent, Tool

from journalism_rag.config import Settings
from journalism_rag.mcp_web import (
    TAVILY_EXTRACT_TOOL,
    TAVILY_SEARCH_TOOL,
    TavilyMCPClient,
    TavilyMCPConfigurationError,
    TavilyMCPResponseError,
    map_tavily_result,
)


RETRIEVED_AT = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)


class Phase6MCPWebTests(unittest.TestCase):
    def test_maps_search_results_and_deduplicates_urls(self):
        result = SimpleNamespace(
            is_error=False,
            structured_content={
                "results": [
                    {
                        "title": "First report",
                        "url": "https://example.com/story?utm_source=x",
                        "content": "First excerpt",
                        "published_date": "2026-09-09T10:00:00Z",
                    },
                    {
                        "title": "Duplicate report",
                        "url": "https://EXAMPLE.com/story",
                        "content": "Duplicate excerpt",
                    },
                ]
            },
        )

        mapped = map_tavily_result(result, retrieved_at=RETRIEVED_AT)

        self.assertEqual(len(mapped), 1)
        self.assertEqual(mapped[0].title, "First report")
        self.assertEqual(mapped[0].url, "https://example.com/story")
        self.assertEqual(mapped[0].published_at.year, 2026)
        self.assertEqual(mapped[0].retrieved_at, RETRIEVED_AT)

    def test_maps_json_text_and_extract_fallback_title(self):
        result = SimpleNamespace(
            is_error=False,
            structured_content=None,
            content=[
                TextContent(
                    type="text",
                    text='{"results": [{"url": "https://example.com/page", "raw_content": "Page text"}]}',
                )
            ],
        )

        mapped = map_tavily_result(result, retrieved_at=RETRIEVED_AT)

        self.assertEqual(mapped[0].title, "example.com")
        self.assertEqual(mapped[0].excerpt, "Page text")

    def test_rejects_errors_and_empty_payloads(self):
        with self.assertRaises(TavilyMCPResponseError):
            map_tavily_result(SimpleNamespace(is_error=True))
        with self.assertRaises(TavilyMCPResponseError):
            map_tavily_result(SimpleNamespace(is_error=False, structured_content={}))

    def test_only_supported_tavily_tools_are_exposed(self):
        definitions = TavilyMCPClient._tool_definitions(
            [
                Tool(
                    name=TAVILY_SEARCH_TOOL,
                    description="Search",
                    inputSchema={"type": "object"},
                ),
                Tool(
                    name=TAVILY_EXTRACT_TOOL,
                    description="Extract",
                    inputSchema={"type": "object"},
                ),
                Tool(name="tavily_crawl", description="Crawl", inputSchema={"type": "object"}),
            ]
        )
        self.assertEqual({item.name for item in definitions}, {TAVILY_SEARCH_TOOL, TAVILY_EXTRACT_TOOL})

    def test_missing_supported_tool_is_rejected(self):
        with self.assertRaises(TavilyMCPResponseError):
            TavilyMCPClient._tool_definitions(
                [Tool(name=TAVILY_SEARCH_TOOL, description="Search", inputSchema={"type": "object"})]
            )

    def test_client_requires_api_key_and_keeps_url_secret_in_memory(self):
        with self.assertRaises(TavilyMCPConfigurationError):
            TavilyMCPClient(Settings())
        client = TavilyMCPClient(Settings(tavily_api_key="test-secret"))
        self.assertIn("tavilyApiKey=test-secret", client._endpoint())


if __name__ == "__main__":
    unittest.main()
