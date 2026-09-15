"""Small Tavily MCP client and response adapter for live research briefs."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from .config import Settings
from .llm import LLMToolValidationError, ToolDefinition
from .tool_validation import validate_tool_definition
from .web_sources import WebEvidence, deduplicate_web_evidence


TAVILY_SEARCH_TOOL = "tavily_search"
TAVILY_EXTRACT_TOOL = "tavily_extract"
TAVILY_SCHEMA_VERSION = "tavily-mcp-1"
_SUPPORTED_TOOLS = frozenset({TAVILY_SEARCH_TOOL, TAVILY_EXTRACT_TOOL})


class TavilyMCPError(RuntimeError):
    """Base class for expected Tavily MCP failures."""


class TavilyMCPConfigurationError(TavilyMCPError):
    """Raised when live-search configuration is missing or invalid."""


class TavilyMCPResponseError(TavilyMCPError):
    """Raised when Tavily returns an unusable MCP result."""


def _parse_published_at(value: Any) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _records_from_payload(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        if isinstance(payload.get("results"), list):
            return [item for item in payload["results"] if isinstance(item, Mapping)]
        if isinstance(payload.get("result"), (Mapping, list)):
            return _records_from_payload(payload["result"])
        if isinstance(payload.get("url"), str):
            return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, Mapping)]
    return []


def map_tavily_result(
    result: Any,
    *,
    retrieved_at: datetime | None = None,
) -> tuple[WebEvidence, ...]:
    """Convert a Tavily MCP result into normalized, bounded web evidence."""

    if getattr(result, "is_error", False):
        raise TavilyMCPResponseError("Tavily MCP tool returned an error")
    payload = getattr(result, "structured_content", None)
    if payload is None and isinstance(result, Mapping):
        payload = result.get("structuredContent") or result.get("structured_content")
    if payload is None:
        text_items = []
        for item in getattr(result, "content", ()) or ():
            text = getattr(item, "text", None)
            if isinstance(text, str):
                text_items.append(text)
        for text in text_items:
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            break
    records = _records_from_payload(payload)
    if not records:
        raise TavilyMCPResponseError("Tavily MCP result contained no web results")
    fetched_at = retrieved_at or datetime.now(timezone.utc)
    evidence: list[WebEvidence] = []
    for record in records:
        url = record.get("url")
        excerpt = record.get("raw_content") or record.get("content")
        if not isinstance(url, str) or not isinstance(excerpt, str) or not excerpt.strip():
            continue
        title = record.get("title")
        if not isinstance(title, str) or not title.strip():
            title = urlsplit(url).hostname or "Web source"
        try:
            evidence.append(
                WebEvidence(
                    title=title,
                    url=url,
                    excerpt=excerpt,
                    retrieved_at=fetched_at,
                    published_at=_parse_published_at(record.get("published_date")),
                )
            )
        except ValueError as exc:
            raise TavilyMCPResponseError("Tavily MCP result contained invalid web evidence") from exc
    if not evidence:
        raise TavilyMCPResponseError("Tavily MCP result contained no usable web results")
    return deduplicate_web_evidence(evidence)


class TavilyMCPClient:
    """Synchronous facade over the official MCP streamable-HTTP client."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.settings.validate()
        if not self.settings.tavily_api_key:
            raise TavilyMCPConfigurationError("TAVILY_API_KEY is required for live research")

    def discover_tools(self) -> tuple[ToolDefinition, ...]:
        """Discover Tavily tools and expose only search and extraction."""

        return self._run(self._discover_tools())

    def search(self, query: str, *, max_results: int = 5) -> tuple[WebEvidence, ...]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query cannot be empty")
        if not 1 <= max_results <= 10:
            raise ValueError("max_results must be between 1 and 10")
        result = self._run(
            self._call_tool(
                TAVILY_SEARCH_TOOL,
                {"query": query.strip(), "max_results": max_results},
            )
        )
        return map_tavily_result(result)

    def extract(self, urls: Sequence[str]) -> tuple[WebEvidence, ...]:
        clean_urls = tuple(url.strip() for url in urls if isinstance(url, str) and url.strip())
        if not clean_urls:
            raise ValueError("urls cannot be empty")
        result = self._run(
            self._call_tool(
                TAVILY_EXTRACT_TOOL,
                {"urls": list(clean_urls), "format": "text"},
            )
        )
        return map_tavily_result(result)

    async def _discover_tools(self) -> tuple[ToolDefinition, ...]:
        async with streamable_http_client(self._endpoint()) as streams:
            async with ClientSession(
                *streams, read_timeout_seconds=self.settings.tavily_timeout_seconds
            ) as session:
                await session.initialize()
                listed = await session.list_tools()
        return self._tool_definitions(listed.tools)

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if name not in _SUPPORTED_TOOLS:
            raise TavilyMCPConfigurationError(f"Unsupported Tavily tool: {name}")
        async with streamable_http_client(self._endpoint()) as streams:
            async with ClientSession(
                *streams, read_timeout_seconds=self.settings.tavily_timeout_seconds
            ) as session:
                await session.initialize()
                listed = await session.list_tools()
                available = {tool.name for tool in listed.tools}
                if name not in available:
                    raise TavilyMCPResponseError(f"Tavily server does not provide {name}")
                return await session.call_tool(
                    name,
                    arguments=arguments,
                    read_timeout_seconds=self.settings.tavily_timeout_seconds,
                )

    @staticmethod
    def _tool_definitions(tools: Sequence[Any]) -> tuple[ToolDefinition, ...]:
        definitions: list[ToolDefinition] = []
        for tool in tools:
            if tool.name not in _SUPPORTED_TOOLS:
                continue
            output_schema = tool.output_schema or {"type": "object"}
            definition = ToolDefinition(
                name=tool.name,
                schema_version=TAVILY_SCHEMA_VERSION,
                description=tool.description or tool.name,
                input_schema=tool.input_schema,
                output_schema=output_schema,
            )
            try:
                validate_tool_definition(definition)
            except LLMToolValidationError as exc:
                raise TavilyMCPResponseError(
                    f"Tavily tool {tool.name!r} has an invalid schema"
                ) from exc
            definitions.append(definition)
        names = {definition.name for definition in definitions}
        if names != _SUPPORTED_TOOLS:
            missing = sorted(_SUPPORTED_TOOLS - names)
            raise TavilyMCPResponseError(
                f"Tavily server is missing supported tools: {', '.join(missing)}"
            )
        return tuple(definitions)

    def _endpoint(self) -> str:
        parsed = urlsplit(self.settings.tavily_mcp_url)
        query = [(key, value) for key, value in parse_qsl(parsed.query, keep_blank_values=True)]
        query = [(key, value) for key, value in query if key.casefold() != "tavilyapikey"]
        query.append(("tavilyApiKey", self.settings.tavily_api_key or ""))
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query), ""))

    @staticmethod
    def _run(coroutine: Any) -> Any:
        try:
            return asyncio.run(coroutine)
        except TavilyMCPError:
            raise
        except Exception as exc:
            raise TavilyMCPError("Tavily MCP request failed") from exc
