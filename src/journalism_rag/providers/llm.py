"""DeepSeek and Ollama adapters using the OpenAI-compatible Chat API."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import replace
from typing import Any, Mapping, Sequence

import requests

from ..config import Settings
from ..llm import (
    ChatModel,
    GenerationOptions,
    GenerationResult,
    GenerationStreamEvent,
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMRequestError,
    LLMStreamError,
    LLMStreamProtocolError,
    LLMStreamingUnsupportedError,
    LLMToolUnsupportedError,
    LLMToolResponseError,
    LLMToolValidationError,
    ToolCall,
    ToolDefinition,
    ToolExchange,
)
from ..tool_validation import (
    validate_tool_arguments,
    validate_tool_definition,
    validate_tool_result,
)


class OpenAICompatibleChatModel(ChatModel):
    """Small dependency-light adapter for OpenAI-compatible endpoints."""

    provider = "openai-compatible"
    supports_tool_calling = False

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str | None,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        input_cost_per_million: float | None = None,
        output_cost_per_million: float | None = None,
        session: requests.Session | None = None,
    ):
        if not model.strip():
            raise LLMConfigurationError("LLM model cannot be empty")
        if not base_url.startswith(("http://", "https://")):
            raise LLMConfigurationError("LLM base URL must start with http:// or https://")
        if timeout_seconds <= 0:
            raise LLMConfigurationError("LLM timeout must be greater than zero")
        if max_retries < 0:
            raise LLMConfigurationError("LLM max_retries cannot be negative")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million
        self.session = session or requests.Session()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        options: GenerationOptions | None = None,
        *,
        tools: Sequence[ToolDefinition] | None = None,
        tool_history: Sequence[ToolExchange] | None = None,
    ) -> GenerationResult:
        if not messages:
            raise ValueError("messages cannot be empty")
        if (tools or tool_history) and not self.supports_tool_calling:
            raise LLMToolUnsupportedError(
                f"{self.provider} does not have verified tool-calling support"
            )
        generation_options = options or GenerationOptions()
        if generation_options.max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")
        definitions_by_name: dict[str, ToolDefinition] = {}
        for tool in tools or ():
            validate_tool_definition(tool)
            if tool.name in definitions_by_name:
                raise LLMToolValidationError(
                    f"Duplicate tool definition name {tool.name!r}"
                )
            definitions_by_name[tool.name] = tool
        request_messages: list[dict[str, Any]] = [dict(message) for message in messages]
        for exchange in tool_history or ():
            request_messages.extend(
                self._tool_exchange_messages(exchange, definitions_by_name)
            )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": request_messages,
            "max_tokens": generation_options.max_tokens,
            "temperature": generation_options.temperature,
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": dict(tool.input_schema),
                    },
                }
                for tool in tools
            ]
        if generation_options.top_p is not None:
            payload["top_p"] = generation_options.top_p
        if generation_options.thinking is not None:
            payload["thinking"] = {"type": generation_options.thinking}
        if generation_options.reasoning_effort is not None:
            payload["reasoning_effort"] = generation_options.reasoning_effort

        started = time.monotonic()
        endpoint = f"{self.base_url}/chat/completions"
        response = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(
                    endpoint,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout_seconds,
                )
            except requests.RequestException as exc:
                if attempt >= self.max_retries:
                    raise LLMRequestError(f"LLM request failed after retries: {exc}") from exc
                time.sleep(min(2**attempt, 8))
                continue
            if response.status_code in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                time.sleep(min(2**attempt, 8))
                continue
            break

        assert response is not None
        if response.status_code in {401, 403}:
            raise LLMAuthenticationError(f"{self.provider} rejected the API credentials")
        if response.status_code >= 400:
            detail = response.text[:500]
            raise LLMRequestError(f"{self.provider} returned HTTP {response.status_code}: {detail}")
        try:
            body = response.json()
            choice = body["choices"][0]
            message = choice["message"]
            if not isinstance(message, Mapping):
                raise TypeError("message must be an object")
            text = message.get("content")
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMRequestError("LLM response did not contain choices[0].message") from exc
        finish_reason = choice.get("finish_reason")
        tool_calls = self._parse_tool_calls(
            message.get("tool_calls"), definitions_by_name
        )
        if finish_reason == "tool_calls" and not tool_calls:
            raise LLMToolResponseError(
                "LLM response finished for tool calls but contained no valid tool calls"
            )
        if tool_calls and (text is None or (isinstance(text, str) and not text.strip())):
            text = ""
        elif not isinstance(text, str) or not text.strip():
            raise LLMRequestError(
                "LLM returned empty message content"
                + (f" (finish_reason={finish_reason})" if finish_reason else "")
                + "; increase max_tokens or adjust the provider reasoning settings"
            )
        usage = body.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        estimated_cost = None
        if (
            prompt_tokens is not None
            and completion_tokens is not None
            and self.input_cost_per_million is not None
            and self.output_cost_per_million is not None
        ):
            estimated_cost = (
                (prompt_tokens / 1_000_000) * self.input_cost_per_million
                + (completion_tokens / 1_000_000) * self.output_cost_per_million
            )
        return GenerationResult(
            text=str(text),
            provider=self.provider,
            model=str(body.get("model", self.model)),
            finish_reason=choice.get("finish_reason"),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=usage.get("total_tokens"),
            latency_seconds=time.monotonic() - started,
            estimated_cost_usd=estimated_cost,
            tool_calls=tool_calls,
        )

    def stream(
        self,
        messages: Sequence[Mapping[str, str]],
        options: GenerationOptions | None = None,
        *,
        tools: Sequence[ToolDefinition] | None = None,
        tool_history: Sequence[ToolExchange] | None = None,
    ) -> Iterator[GenerationStreamEvent]:
        """Stream OpenAI-compatible SSE chunks as normalized events."""

        if not self.supports_streaming:
            raise LLMStreamingUnsupportedError(
                f"{self.provider} does not have verified streaming support"
            )
        if not messages:
            raise ValueError("messages cannot be empty")
        if tools or tool_history:
            raise LLMStreamingUnsupportedError(
                "streaming tool calls are not implemented; use generate()"
            )
        generation_options = options or GenerationOptions()
        if generation_options.max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "max_tokens": generation_options.max_tokens,
            "temperature": generation_options.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if generation_options.top_p is not None:
            payload["top_p"] = generation_options.top_p
        if generation_options.thinking is not None:
            payload["thinking"] = {"type": generation_options.thinking}
        if generation_options.reasoning_effort is not None:
            payload["reasoning_effort"] = generation_options.reasoning_effort

        started = time.monotonic()
        endpoint = f"{self.base_url}/chat/completions"
        try:
            response = self.session.post(
                endpoint,
                headers=self._headers(),
                json=payload,
                timeout=self.timeout_seconds,
                stream=True,
            )
        except requests.RequestException as exc:
            yield GenerationStreamEvent.failed(LLMStreamError(f"stream request failed: {exc}"))
            return

        try:
            if response.status_code >= 400:
                detail = response.text[:500]
                yield GenerationStreamEvent.failed(
                    LLMStreamError(
                        f"{self.provider} streaming request returned HTTP "
                        f"{response.status_code}: {detail}"
                    )
                )
                return
            pieces: list[str] = []
            finish_reason: str | None = None
            usage: Mapping[str, Any] = {}
            saw_done = False
            # Split the byte stream before decoding. ``requests`` can otherwise
            # decode text/event-stream as Latin-1, where UTF-8 bytes such as
            # 0x85 become Unicode line separators and split Greek JSON strings.
            for raw_line in response.iter_lines(decode_unicode=False):
                if isinstance(raw_line, bytes):
                    raw_line = raw_line.decode("utf-8", errors="replace")
                line = str(raw_line).strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    saw_done = True
                    break
                try:
                    body = json.loads(data)
                except (json.JSONDecodeError, TypeError):
                    yield GenerationStreamEvent.failed(
                        LLMStreamProtocolError("stream event was not valid OpenAI JSON")
                    )
                    return
                if not isinstance(body, Mapping):
                    yield GenerationStreamEvent.failed(
                        LLMStreamProtocolError("stream event must be a JSON object")
                    )
                    return
                event_usage = body.get("usage")
                if isinstance(event_usage, Mapping):
                    usage = event_usage
                choices = body.get("choices")
                if not isinstance(choices, list):
                    yield GenerationStreamEvent.failed(
                        LLMStreamProtocolError("stream event choices must be a list")
                    )
                    return
                if not choices:
                    if isinstance(event_usage, Mapping):
                        continue
                    yield GenerationStreamEvent.failed(
                        LLMStreamProtocolError(
                            "stream event had neither a choice nor usage metadata"
                        )
                    )
                    return
                choice = choices[0]
                if not isinstance(choice, Mapping):
                    yield GenerationStreamEvent.failed(
                        LLMStreamProtocolError("stream choice must be a JSON object")
                    )
                    return
                delta = choice.get("delta") or {}
                if not isinstance(delta, Mapping):
                    yield GenerationStreamEvent.failed(
                        LLMStreamProtocolError("stream delta must be a JSON object")
                    )
                    return
                content = delta.get("content")
                if content is not None and not isinstance(content, str):
                    yield GenerationStreamEvent.failed(
                        LLMStreamProtocolError("stream delta content must be a string")
                    )
                    return
                if content:
                    pieces.append(content)
                    yield GenerationStreamEvent.delta(content)
                finish_reason = choice.get("finish_reason") or finish_reason
            if not saw_done:
                yield GenerationStreamEvent.failed(
                    LLMStreamProtocolError("stream ended before the [DONE] event")
                )
                return
            text = "".join(pieces)
            if not text.strip():
                yield GenerationStreamEvent.failed(
                    LLMStreamProtocolError("stream completed without message content")
                )
                return
            result = GenerationResult(
                text=text,
                provider=self.provider,
                model=self.model,
                finish_reason=finish_reason,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
                latency_seconds=time.monotonic() - started,
            )
            yield GenerationStreamEvent.completed(result)
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()

    @staticmethod
    def _parse_tool_calls(
        raw_tool_calls: Any,
        definitions_by_name: Mapping[str, ToolDefinition],
    ) -> tuple[ToolCall, ...]:
        if raw_tool_calls is None:
            return ()
        if not isinstance(raw_tool_calls, list):
            raise LLMToolResponseError("LLM tool_calls must be a list")

        parsed_calls: list[ToolCall] = []
        for index, raw_call in enumerate(raw_tool_calls):
            try:
                call_id = raw_call["id"]
                call_type = raw_call["type"]
                function = raw_call["function"]
                name = function["name"]
                raw_arguments = function["arguments"]
            except (KeyError, IndexError, TypeError) as exc:
                raise LLMToolResponseError(
                    f"LLM tool call at index {index} is missing required fields"
                ) from exc
            if call_type != "function":
                raise LLMToolResponseError(
                    f"LLM tool call at index {index} has unsupported type {call_type!r}"
                )
            if not isinstance(call_id, str) or not call_id.strip():
                raise LLMToolResponseError(
                    f"LLM tool call at index {index} has an invalid id"
                )
            if not isinstance(name, str) or not name.strip():
                raise LLMToolResponseError(
                    f"LLM tool call at index {index} has an invalid function name"
                )
            if not isinstance(raw_arguments, str):
                raise LLMToolResponseError(
                    f"LLM tool call {call_id!r} arguments must be a JSON string"
                )
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                raise LLMToolResponseError(
                    f"LLM tool call {call_id!r} arguments are not valid JSON"
                ) from exc
            if not isinstance(arguments, dict):
                raise LLMToolResponseError(
                    f"LLM tool call {call_id!r} arguments must decode to an object"
                )
            call = ToolCall(call_id=call_id, name=name, arguments=arguments)
            definition = definitions_by_name.get(call.name)
            if definition is None:
                raise LLMToolValidationError(
                    f"LLM requested unknown tool {call.name!r}"
                )
            validate_tool_arguments(definition, call)
            parsed_calls.append(call)
        return tuple(parsed_calls)

    @staticmethod
    def _tool_exchange_messages(
        exchange: ToolExchange,
        definitions_by_name: Mapping[str, ToolDefinition],
    ) -> list[dict[str, Any]]:
        if not exchange.calls:
            raise LLMToolResponseError("Tool exchange must contain at least one call")
        calls_by_id = {call.call_id: call for call in exchange.calls}
        if len(calls_by_id) != len(exchange.calls):
            raise LLMToolResponseError("Tool exchange contains duplicate call ids")

        assistant_calls: list[dict[str, Any]] = []
        for call in exchange.calls:
            definition = definitions_by_name.get(call.name)
            if definition is None:
                raise LLMToolValidationError(
                    f"Tool history contains unknown tool {call.name!r}"
                )
            validate_tool_arguments(definition, call)
            try:
                arguments = json.dumps(
                    dict(call.arguments), ensure_ascii=False, sort_keys=True
                )
            except (TypeError, ValueError) as exc:
                raise LLMToolResponseError(
                    f"Tool call {call.call_id!r} arguments are not JSON serializable"
                ) from exc
            assistant_calls.append(
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": arguments},
                }
            )

        messages: list[dict[str, Any]] = [
            {"role": "assistant", "content": None, "tool_calls": assistant_calls}
        ]
        seen_result_ids: set[str] = set()
        for result in exchange.results:
            call = calls_by_id.get(result.call_id)
            if call is None or call.name != result.name:
                raise LLMToolResponseError(
                    f"Tool result {result.call_id!r} does not match an exchange call"
                )
            if result.call_id in seen_result_ids:
                raise LLMToolResponseError(
                    f"Tool exchange contains duplicate result for {result.call_id!r}"
                )
            seen_result_ids.add(result.call_id)
            definition = definitions_by_name[call.name]
            validate_tool_result(definition, result)
            try:
                content = json.dumps(
                    {"is_error": result.is_error, "result": dict(result.content)},
                    ensure_ascii=False,
                    sort_keys=True,
                )
            except (TypeError, ValueError) as exc:
                raise LLMToolResponseError(
                    f"Tool result {result.call_id!r} is not JSON serializable"
                ) from exc
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": result.call_id,
                    "content": content,
                }
            )
        if seen_result_ids != set(calls_by_id):
            missing = sorted(set(calls_by_id) - seen_result_ids)
            raise LLMToolResponseError(
                f"Tool exchange is missing results for call ids: {', '.join(missing)}"
            )
        return messages


class DeepSeekChatModel(OpenAICompatibleChatModel):
    """Hosted DeepSeek Chat Completions adapter."""

    provider = "deepseek"
    supports_tool_calling = True
    supports_streaming = True
    # Current published DeepSeek V4.1-Flash off-peak rates, in USD per 1M
    # tokens. Peak rates are higher; the adapter reports an estimate only.
    _PRICES = {
        "deepseek-flash": (0.15, 0.60),
        # Keep legacy identifiers usable while they are still accepted by the
        # API; DeepSeek routes the retired flash name to V4.1-Flash.
        "deepseek-v4-flash": (0.15, 0.60),
        "deepseek-v4-pro": (0.15, 0.60),
    }

    def __init__(self, settings: Settings, *, session: requests.Session | None = None):
        if not settings.deepseek_api_key:
            raise LLMConfigurationError("DEEPSEEK_API_KEY is required for the DeepSeek provider")
        super().__init__(
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            api_key=settings.deepseek_api_key,
            timeout_seconds=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
            input_cost_per_million=self._PRICES.get(settings.llm_model, (None, None))[0],
            output_cost_per_million=self._PRICES.get(settings.llm_model, (None, None))[1],
            session=session,
        )
        self.default_thinking = settings.deepseek_thinking

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        options: GenerationOptions | None = None,
        *,
        tools: Sequence[ToolDefinition] | None = None,
        tool_history: Sequence[ToolExchange] | None = None,
    ) -> GenerationResult:
        """Use the configured DeepSeek thinking mode unless a call overrides it."""

        generation_options = options or GenerationOptions()
        if generation_options.thinking is None:
            generation_options = replace(
                generation_options,
                thinking=self.default_thinking,
            )
        return super().generate(
            messages,
            generation_options,
            tools=tools,
            tool_history=tool_history,
        )

    def stream(
        self,
        messages: Sequence[Mapping[str, str]],
        options: GenerationOptions | None = None,
        *,
        tools: Sequence[ToolDefinition] | None = None,
        tool_history: Sequence[ToolExchange] | None = None,
    ) -> Iterator[GenerationStreamEvent]:
        """Apply the configured DeepSeek thinking mode to streamed requests."""

        generation_options = options or GenerationOptions()
        if generation_options.thinking is None:
            generation_options = replace(
                generation_options,
                thinking=self.default_thinking,
            )
        return super().stream(
            messages,
            generation_options,
            tools=tools,
            tool_history=tool_history,
        )


class OllamaChatModel(OpenAICompatibleChatModel):
    """Explicit local Ollama adapter; it never downloads models automatically."""

    # Ollama's OpenAI-compatible endpoint is verified for the selected local
    # model in Phase 8.  Tool calls remain non-streaming and use ``generate``.
    supports_tool_calling = True
    supports_streaming = True

    provider = "ollama"

    def __init__(self, settings: Settings, *, session: requests.Session | None = None):
        super().__init__(
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            api_key="ollama",
            timeout_seconds=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
            session=session,
        )


def create_chat_model(settings: Settings, *, session: requests.Session | None = None) -> ChatModel:
    """Create exactly the provider selected in settings; never silently fallback."""

    settings.validate()
    if settings.llm_provider == "deepseek":
        return DeepSeekChatModel(settings, session=session)
    if settings.llm_provider == "ollama":
        return OllamaChatModel(settings, session=session)
    raise LLMConfigurationError(f"Unsupported LLM provider: {settings.llm_provider}")
