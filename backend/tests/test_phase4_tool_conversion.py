"""Phase 4 tests: tool definition conversion (ToolInterface → LLMToolDefinition)."""

from __future__ import annotations

import json

from app.core.identifiers import generate_id
from app.llm import (
    build_assistant_tool_call_message,
    build_tool_result_message,
    tool_info_to_definition,
    tool_to_definition,
    tools_to_definitions,
)
from app.llm.models import LLMToolCall, LLMToolDefinition, MessageRole
from app.tools.impl import CalculatorTool
from app.tools.interface import ToolInfo, ToolResult


def test_tool_to_definition_converts_calculator() -> None:
    tool = CalculatorTool()
    defn = tool_to_definition(tool)
    assert isinstance(defn, LLMToolDefinition)
    assert defn.name == "calculator"
    assert "arithmetic" in defn.description.lower() or "calc" in defn.description.lower()
    assert defn.parameters["type"] == "object"
    assert "expression" in defn.parameters["properties"]


def test_tool_info_to_definition_works_without_instance() -> None:
    info = ToolInfo(
        tool_id="dummy",
        name="dummy_tool",
        description="A dummy tool.",
        input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
    )
    defn = tool_info_to_definition(info)
    assert defn.name == "dummy_tool"
    assert defn.description == "A dummy tool."
    assert defn.parameters == info.input_schema


def test_tool_info_to_definition_defaults_empty_schema() -> None:
    info = ToolInfo(tool_id="t", name="t", description="d", input_schema={})
    defn = tool_info_to_definition(info)
    assert defn.parameters == {"type": "object", "properties": {}}


def test_tools_to_definitions_converts_list() -> None:
    tools = [CalculatorTool(), CalculatorTool()]
    defs = tools_to_definitions(tools)
    assert len(defs) == 2
    assert all(isinstance(d, LLMToolDefinition) for d in defs)
    assert defs[0].name == "calculator"


def test_tools_to_definitions_empty_list() -> None:
    assert tools_to_definitions([]) == []


def test_build_assistant_tool_call_message_has_tool_calls() -> None:
    calls = [
        LLMToolCall(id="call_1", name="calculator", arguments={"expression": "2+2"}),
    ]
    msg = build_assistant_tool_call_message("Let me calculate.", calls)
    assert msg.role is MessageRole.ASSISTANT
    assert msg.content == "Let me calculate."
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].name == "calculator"
    assert msg.tool_calls[0].arguments == {"expression": "2+2"}


def test_build_tool_result_message_serializes_success() -> None:
    call = LLMToolCall(id="call_1", name="calculator", arguments={"expression": "2+2"})
    result = ToolResult(
        invocation_id=generate_id("inv"),
        tool_call_id="call_1",
        name="calculator",
        success=True,
        output={"result": 4},
    )
    msg = build_tool_result_message(call, result)
    assert msg.role is MessageRole.TOOL
    assert msg.tool_call_id == "call_1"
    assert msg.name == "calculator"
    payload = json.loads(msg.content)
    assert payload["success"] is True
    assert payload["output"] == {"result": 4}
    assert "error" not in payload


def test_build_tool_result_message_includes_error() -> None:
    call = LLMToolCall(id="call_2", name="calculator", arguments={"expression": "1/0"})
    result = ToolResult(
        invocation_id=generate_id("inv"),
        tool_call_id="call_2",
        name="calculator",
        success=False,
        error="Division by zero.",
    )
    msg = build_tool_result_message(call, result)
    payload = json.loads(msg.content)
    assert payload["success"] is False
    assert payload["error"] == "Division by zero."
