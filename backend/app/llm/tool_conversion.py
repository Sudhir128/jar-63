"""Tool definition conversion: ToolInterface → LLMToolDefinition.

Converts JAR-63 tools (registered in the :class:`ToolRegistry`) into the
provider-independent :class:`LLMToolDefinition` format that the LLM
abstraction layer expects. Provider implementations then map
:class:`LLMToolDefinition` to their native tool-calling format (e.g. Ollama's
``{"type": "function", "function": {...}}`` or OpenAI's function schema).

This conversion happens at the boundary between the tool layer and the LLM
layer. The LLM never sees internal tool implementation details — only the
public name, description, and argument JSON Schema.

Also provides helpers for building the tool-result conversation messages that
are fed back to the LLM after a tool executes.
"""

from __future__ import annotations

import json
from typing import Any

from app.llm.models import LLMMessage, LLMToolCall, LLMToolDefinition, MessageRole
from app.tools.interface import ToolInterface, ToolResult

__all__ = [
    "tool_to_definition",
    "tools_to_definitions",
    "tool_info_to_definition",
    "build_tool_result_message",
    "build_assistant_tool_call_message",
]


def tool_to_definition(tool: ToolInterface) -> LLMToolDefinition:
    """Convert a registered tool into an :class:`LLMToolDefinition`."""
    return tool_info_to_definition(tool.info)


def tool_info_to_definition(info: Any) -> LLMToolDefinition:
    """Convert a :class:`ToolInfo` into an :class:`LLMToolDefinition`.

    This operates on the static tool metadata, so it can be used without a
    live tool instance (e.g. when building definitions from cached info).
    """
    return LLMToolDefinition(
        name=info.name,
        description=info.description,
        parameters=info.input_schema or {"type": "object", "properties": {}},
    )


def tools_to_definitions(tools: list[ToolInterface]) -> list[LLMToolDefinition]:
    """Convert a list of registered tools into LLM tool definitions."""
    return [tool_to_definition(t) for t in tools]


def build_assistant_tool_call_message(
    content: str, tool_calls: list[LLMToolCall]
) -> LLMMessage:
    """Build the assistant message carrying tool calls for the conversation.

    This message is appended to the conversation history so the LLM knows
    which tool calls it requested.
    """
    return LLMMessage(
        role=MessageRole.ASSISTANT,
        content=content,
        tool_calls=list(tool_calls),
    )


def build_tool_result_message(tool_call: LLMToolCall, result: ToolResult) -> LLMMessage:
    """Build a tool-role message feeding a tool result back to the LLM.

    The result is serialized as JSON (the provider-agnostic format). The
    ``tool_call_id`` correlates this result with the original tool call so the
    provider can match them in multi-turn conversations.
    """
    payload: dict[str, Any] = {
        "success": result.success,
        "output": result.output,
    }
    if result.error:
        payload["error"] = result.error
    return LLMMessage(
        role=MessageRole.TOOL,
        content=json.dumps(payload, default=str),
        tool_call_id=tool_call.id,
        name=tool_call.name,
    )
