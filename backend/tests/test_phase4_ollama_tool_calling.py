"""Phase 4 tests: Ollama native tool-calling (mocked HTTP, fully offline)."""

from __future__ import annotations

import json

import httpx

from app.llm.models import (
    FinishReason,
    LLMMessage,
    LLMRequest,
    LLMToolCall,
    MessageRole,
)
from app.llm.providers.ollama import OllamaClient
from app.llm.tool_conversion import tool_to_definition
from app.tools.impl import CalculatorTool

CALC_DEF = tool_to_definition(CalculatorTool())


def _ollama_tool_call_response(
    *,
    tool_name: str = "calculator",
    arguments: dict | None = None,
    call_id: str | None = None,
    content: str = "",
) -> dict:
    """Build an Ollama chat response that requests a tool call."""
    func: dict = {"name": tool_name}
    if arguments is not None:
        func["arguments"] = arguments
    else:
        func["arguments"] = {"expression": "2 + 2"}
    tc: dict = {"function": func}
    if call_id:
        tc["id"] = call_id
    return {
        "model": "qwen2.5-coder:7b",
        "message": {"role": "assistant", "content": content, "tool_calls": [tc]},
        "done": True,
        "prompt_eval_count": 12,
        "eval_count": 5,
    }


def _make_ollama_client(chat_response: dict) -> tuple[OllamaClient, list[httpx.Request]]:
    """Create an OllamaClient with a mocked transport."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={"models": [{"name": "qwen2.5-coder:7b"}]},
            )
        if request.url.path == "/api/chat":
            return httpx.Response(200, json=chat_response)
        return httpx.Response(404, text="not found")

    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(
        base_url="http://localhost:11434", transport=transport, timeout=10.0
    )
    client = OllamaClient(
        base_url="http://localhost:11434",
        timeout=10.0,
        http_client=http_client,
    )
    return client, requests


async def test_ollama_sends_tool_definitions_in_payload() -> None:
    resp_data = _ollama_tool_call_response()
    client, requests = _make_ollama_client(resp_data)
    try:
        await client.generate(
            LLMRequest(
                model="qwen2.5-coder:7b",
                messages=[LLMMessage(role=MessageRole.USER, content="What is 2+2?")],
                tools=[CALC_DEF],
            )
        )
        assert len(requests) >= 1
        chat_req = requests[-1]
        payload = json.loads(chat_req.content)
        assert "tools" in payload
        assert payload["tools"][0]["type"] == "function"
        assert payload["tools"][0]["function"]["name"] == "calculator"
        assert "parameters" in payload["tools"][0]["function"]
    finally:
        await client.close()


async def test_ollama_parses_single_tool_call() -> None:
    resp_data = _ollama_tool_call_response(
        arguments={"expression": "3 * 7"},
        call_id="call_abc",
        content="Let me calculate that.",
    )
    client, _ = _make_ollama_client(resp_data)
    try:
        resp = await client.generate(
            LLMRequest(
                model="qwen2.5-coder:7b",
                messages=[LLMMessage(role=MessageRole.USER, content="What is 3*7?")],
                tools=[CALC_DEF],
            )
        )
        assert resp.has_tool_calls
        assert len(resp.tool_calls) == 1
        tc = resp.tool_calls[0]
        assert tc.name == "calculator"
        assert tc.arguments == {"expression": "3 * 7"}
        assert tc.id == "call_abc"
        assert resp.finish_reason is FinishReason.TOOL_CALLS
        assert resp.content == "Let me calculate that."
    finally:
        await client.close()


async def test_ollama_parses_multiple_tool_calls() -> None:
    resp_data = {
        "model": "qwen2.5-coder:7b",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {"name": "calculator", "arguments": {"expression": "1+1"}},
                },
                {
                    "id": "c2",
                    "function": {"name": "calculator", "arguments": {"expression": "2+2"}},
                },
            ],
        },
        "done": True,
    }
    client, _ = _make_ollama_client(resp_data)
    try:
        resp = await client.generate(
            LLMRequest(
                model="qwen2.5-coder:7b",
                messages=[LLMMessage(role=MessageRole.USER, content="calc two things")],
                tools=[CALC_DEF],
            )
        )
        assert len(resp.tool_calls) == 2
        assert {tc.id for tc in resp.tool_calls} == {"c1", "c2"}
        assert resp.finish_reason is FinishReason.TOOL_CALLS
    finally:
        await client.close()


async def test_ollama_arguments_as_json_string_are_parsed() -> None:
    """Some models return arguments as a JSON string; Ollama provider parses it."""
    resp_data = {
        "model": "qwen2.5-coder:7b",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {"name": "calculator", "arguments": '{"expression": "10 / 2"}'},
                },
            ],
        },
        "done": True,
    }
    client, _ = _make_ollama_client(resp_data)
    try:
        resp = await client.generate(
            LLMRequest(
                model="qwen2.5-coder:7b",
                messages=[LLMMessage(role=MessageRole.USER, content="calc")],
                tools=[CALC_DEF],
            )
        )
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].arguments == {"expression": "10 / 2"}
    finally:
        await client.close()


async def test_ollama_malformed_tool_calls_are_skipped() -> None:
    """Malformed tool call entries are skipped, never reaching the executor."""
    resp_data = {
        "model": "qwen2.5-coder:7b",
        "message": {
            "role": "assistant",
            "content": "I'm not sure.",
            "tool_calls": [
                {"function": {}},  # missing name
                {"id": "c2", "function": {"name": "calculator", "arguments": "not-json"}},
            ],
        },
        "done": True,
    }
    client, _ = _make_ollama_client(resp_data)
    try:
        resp = await client.generate(
            LLMRequest(
                model="qwen2.5-coder:7b",
                messages=[LLMMessage(role=MessageRole.USER, content="calc")],
                tools=[CALC_DEF],
            )
        )
        # The malformed one (bad JSON) becomes {}, the nameless one is skipped.
        valid = [tc for tc in resp.tool_calls if tc.name]
        assert len(valid) == 1
        assert valid[0].arguments == {}
    finally:
        await client.close()


async def test_ollama_no_tools_in_request_omits_tools_field() -> None:
    resp_data = {
        "model": "qwen2.5-coder:7b",
        "message": {"role": "assistant", "content": "Hello!"},
        "done": True,
        "prompt_eval_count": 5,
        "eval_count": 2,
    }
    client, requests = _make_ollama_client(resp_data)
    try:
        await client.generate(
            LLMRequest(
                model="qwen2.5-coder:7b",
                messages=[LLMMessage(role=MessageRole.USER, content="Hi")],
            )
        )
        payload = json.loads(requests[-1].content)
        assert "tools" not in payload
    finally:
        await client.close()


async def test_ollama_assistant_tool_call_message_serialized_in_request() -> None:
    """When sending back an assistant message with tool_calls, the provider
    serializes them in the Ollama tool_call format."""
    resp_data = _ollama_tool_call_response()
    client, requests = _make_ollama_client(resp_data)
    try:
        tool_calls = [LLMToolCall(id="call_1", name="calculator", arguments={"expression": "2+2"})]
        await client.generate(
            LLMRequest(
                model="qwen2.5-coder:7b",
                messages=[
                    LLMMessage(role=MessageRole.USER, content="What is 2+2?"),
                    LLMMessage(
                        role=MessageRole.ASSISTANT,
                        content="",
                        tool_calls=tool_calls,
                    ),
                ],
                tools=[CALC_DEF],
            )
        )
        payload = json.loads(requests[-1].content)
        msgs = payload["messages"]
        assistant_msg = msgs[1]
        assert assistant_msg["role"] == "assistant"
        assert "tool_calls" in assistant_msg
        assert assistant_msg["tool_calls"][0]["function"]["name"] == "calculator"
        assert assistant_msg["tool_calls"][0]["function"]["arguments"] == {"expression": "2+2"}
    finally:
        await client.close()


async def test_ollama_tool_result_message_serialized_in_request() -> None:
    """A tool-role message with tool_call_id is serialized correctly."""
    resp_data = _ollama_tool_call_response()
    client, requests = _make_ollama_client(resp_data)
    try:
        await client.generate(
            LLMRequest(
                model="qwen2.5-coder:7b",
                messages=[
                    LLMMessage(role=MessageRole.USER, content="What is 2+2?"),
                    LLMMessage(
                        role=MessageRole.ASSISTANT,
                        content="",
                        tool_calls=[
                            LLMToolCall(
                                id="call_1", name="calculator", arguments={"expression": "2+2"}
                            )
                        ],
                    ),
                    LLMMessage(
                        role=MessageRole.TOOL,
                        content=json.dumps({"success": True, "output": {"result": 4}}),
                        tool_call_id="call_1",
                        name="calculator",
                    ),
                ],
                tools=[CALC_DEF],
            )
        )
        payload = json.loads(requests[-1].content)
        tool_msg = payload["messages"][2]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "call_1"
    finally:
        await client.close()
