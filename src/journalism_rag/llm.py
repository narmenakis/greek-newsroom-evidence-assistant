"""Provider-neutral chat model contracts and result types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections.abc import Iterator
from typing import Any, Literal, Mapping, Sequence


class LLMError(RuntimeError):
    """Base class for expected provider failures."""


class LLMConfigurationError(LLMError):
    """Raised when a provider is missing required configuration."""


class LLMAuthenticationError(LLMError):
    """Raised when a provider rejects the configured credentials."""


class LLMRequestError(LLMError):
    """Raised for non-retryable provider or response errors."""


class LLMToolError(LLMRequestError):
    """Base class for normalized tool-calling failures."""


class LLMToolUnsupportedError(LLMToolError):
    """Raised when the selected provider cannot use the tool contract."""


class LLMToolResponseError(LLMToolError):
    """Raised when a provider returns a malformed tool-calling response."""


class LLMToolValidationError(LLMToolError):
    """Raised when tool data violates its declared, versioned schema."""


class LLMStreamError(LLMError):
    """Base class for normalized streaming failures."""


class LLMStreamingUnsupportedError(LLMStreamError):
    """Raised when a provider has no verified streaming implementation."""


class LLMStreamProtocolError(LLMStreamError):
    """Raised when a provider emits an invalid or incomplete stream."""


@dataclass(frozen=True)
class GenerationOptions:
    max_tokens: int = 8192
    temperature: float = 0.1
    top_p: float | None = None
    thinking: str | None = None
    reasoning_effort: str | None = None


@dataclass(frozen=True)
class ToolDefinition:
    """Provider-neutral description of a tool available to a chat model.

    The schemas use JSON Schema Draft 2020-12. ``schema_version`` versions the
    individual tool contract independently from the JSON Schema dialect.
    """

    name: str
    schema_version: str
    description: str
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """Structured request made by a model to invoke one named tool."""

    call_id: str
    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class ToolResult:
    """Structured application result associated with a model tool call."""

    call_id: str
    name: str
    content: Mapping[str, Any]
    is_error: bool = False


@dataclass(frozen=True)
class ToolExchange:
    """One assistant tool-call turn and the application's matching results."""

    calls: tuple[ToolCall, ...]
    results: tuple[ToolResult, ...]


@dataclass(frozen=True)
class GenerationResult:
    text: str
    provider: str
    model: str
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    latency_seconds: float
    estimated_cost_usd: float | None = None
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True)
class GenerationStreamEvent:
    """One normalized event emitted by an optional streaming generation call."""

    event_type: Literal["delta", "completed", "error"]
    text: str = ""
    result: GenerationResult | None = None
    error: LLMStreamError | None = None

    def __post_init__(self) -> None:
        if self.event_type == "delta":
            if not self.text:
                raise ValueError("delta stream events require non-empty text")
            if self.result is not None or self.error is not None:
                raise ValueError("delta stream events cannot contain result or error")
        elif self.event_type == "completed":
            if self.result is None:
                raise ValueError("completed stream events require a generation result")
            if self.text or self.error is not None:
                raise ValueError("completed stream events cannot contain text or error")
        elif self.event_type == "error":
            if self.error is None:
                raise ValueError("error stream events require a normalized error")
            if self.text or self.result is not None:
                raise ValueError("error stream events cannot contain text or result")
        else:
            raise ValueError(f"unsupported stream event type: {self.event_type!r}")

    @classmethod
    def delta(cls, text: str) -> "GenerationStreamEvent":
        return cls(event_type="delta", text=text)

    @classmethod
    def completed(cls, result: GenerationResult) -> "GenerationStreamEvent":
        return cls(event_type="completed", result=result)

    @classmethod
    def failed(cls, error: LLMStreamError) -> "GenerationStreamEvent":
        return cls(event_type="error", error=error)


class ChatModel(ABC):
    """Interface implemented by hosted and local chat providers."""

    provider: str
    model: str
    supports_streaming = False

    @abstractmethod
    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        options: GenerationOptions | None = None,
        *,
        tools: Sequence[ToolDefinition] | None = None,
        tool_history: Sequence[ToolExchange] | None = None,
    ) -> GenerationResult:
        raise NotImplementedError

    def stream(
        self,
        messages: Sequence[Mapping[str, str]],
        options: GenerationOptions | None = None,
        *,
        tools: Sequence[ToolDefinition] | None = None,
        tool_history: Sequence[ToolExchange] | None = None,
    ) -> Iterator[GenerationStreamEvent]:
        """Optionally stream normalized events; providers opt in explicitly."""

        del messages, options, tools, tool_history
        raise LLMStreamingUnsupportedError(
            f"{self.provider} does not have verified streaming support"
        )


def options_from_settings(settings: Any) -> GenerationOptions:
    return GenerationOptions(
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
        reasoning_effort=(
            settings.ollama_reasoning_effort
            if settings.llm_provider == "ollama"
            else None
        ),
    )
