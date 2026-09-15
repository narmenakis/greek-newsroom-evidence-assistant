"""Fail-closed validation for versioned tool arguments and results."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from .llm import LLMToolValidationError, ToolCall, ToolDefinition, ToolResult


def validate_tool_definition(definition: ToolDefinition) -> None:
    """Reject incomplete definitions and invalid input or output schemas."""

    if not definition.name.strip():
        raise LLMToolValidationError("Tool name cannot be empty")
    if not definition.schema_version.strip():
        raise LLMToolValidationError(
            f"Tool {definition.name!r} schema version cannot be empty"
        )
    for label, schema in (
        ("input", definition.input_schema),
        ("output", definition.output_schema),
    ):
        if not isinstance(schema, Mapping):
            raise LLMToolValidationError(
                f"Tool {definition.name!r} {label} schema must be an object"
            )
        try:
            Draft202012Validator.check_schema(dict(schema))
        except SchemaError as exc:
            raise LLMToolValidationError(
                f"Tool {definition.name!r} has an invalid {label} schema "
                f"for version {definition.schema_version!r}"
            ) from exc


def validate_tool_arguments(definition: ToolDefinition, call: ToolCall) -> None:
    """Validate one model-generated argument object before tool execution."""

    validate_tool_definition(definition)
    if call.name != definition.name:
        raise LLMToolValidationError(
            f"Tool call {call.call_id!r} names {call.name!r}, not {definition.name!r}"
        )
    _validate_instance(
        definition=definition,
        instance=call.arguments,
        schema=definition.input_schema,
        label="arguments",
    )


def validate_tool_result(definition: ToolDefinition, result: ToolResult) -> None:
    """Validate one tool result before it is returned to the model."""

    validate_tool_definition(definition)
    if result.name != definition.name:
        raise LLMToolValidationError(
            f"Tool result {result.call_id!r} names {result.name!r}, "
            f"not {definition.name!r}"
        )
    _validate_instance(
        definition=definition,
        instance=result.content,
        schema=definition.output_schema,
        label="result",
    )


def _validate_instance(
    *,
    definition: ToolDefinition,
    instance: Mapping[str, Any],
    schema: Mapping[str, Any],
    label: str,
) -> None:
    try:
        Draft202012Validator(dict(schema)).validate(dict(instance))
    except ValidationError as exc:
        path = ".".join(str(part) for part in exc.absolute_path) or "<root>"
        raise LLMToolValidationError(
            f"Tool {definition.name!r} schema version {definition.schema_version!r} "
            f"rejected {label} at {path!r} ({exc.validator} validation)"
        ) from exc
