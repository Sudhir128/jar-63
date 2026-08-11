"""Phase 4 tests: OpenAI-compatible provider tool-calling (mocked, offline)."""

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
from app.llm.providers.openai_compatible import OpenAICompatibleClient
from app.llm.tool_conversion import tool_to_definition
from app.tools.impl import CalculatorTool

CALC_DEF = tool_to_definition(CalculatorTool())


def _openai_tool_call_response(
    *,
    tool_name: str = "calculator",
    arguments: str | None = None,
    call_id: str = "call_1",
    content: str = "",
    finish_reason: str = "tool_calls",
) -> dict:
    if arguments is None:
        arguments = json.dumps({"expression": "2 + 2"})
    return {
        "id": "chatcmpl-1",
        "model": "cloud-coder:32b",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": tool_name, "arguments": arguments},
                        }
                    ],
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _make_openai_client(
    completion_response: dict, *, api_key: str = "test-key"
) -> tuple[OpenAICompatibleClient, list[httpx.Request]]:
    requests: list[httpx.Request] = []
    auth_headers: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        auth_headers.append(request.headers.get("authorization"))
        if request.url.path.endswith("/v1/models"):
            return httpx.Response(200, json={"data": [{"id": "cloud-coder:32b"}]})
        if request.url.path.endswith("/v1/chat/completions"):
            return httpx.Response(200, json=completion_response)
        return httpx.Response(404, text="not found")

    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport, timeout=10.0)
    client = OpenAICompatibleClient(
        base_url="http://cloud.example.com",
        model="cloud-coder:32b",
        api_key=api_key,
        timeout=10.0,
        http_client=http_client,
    )
    return client, requests


async def test_openai_sends_tool_definitions_in_payload() -> None:
    resp_data = _openai_tool_call_response()
    client, requests = _make_openai_client(resp_data)
    try:
        await client.generate(
            LLMRequest(
                model="cloud-coder:32b",
                messages=[LLMMessage(role=MessageRole.USER, content="What is 2+2?")],
                tools=[CALC_DEF],
            )
        )
        payload = json.loads(requests[-1].content)
        assert "tools" in payload
        assert payload["tools"][0]["type"] == "function"
        assert payload["tools"][0]["function"]["name"] == "calculator"
        assert "parameters" in payload["tools"][0]["function"]
    finally:
        await client.close()


async def test_openai_parses_single_tool_call() -> None:
    resp_data = _openai_tool_call_response(
        arguments=json.dumps({"expression": "5 * 6"}),
        call_id="call_xyz",
        content="Let me compute that.",
    )
    client, _ = _make_openai_client(resp_data)
    try:
        resp = await client.generate(
            LLMRequest(
                model="cloud-coder:32b",
                messages=[LLMMessage(role=MessageRole.USER, content="What is 5*6?")],
                tools=[CALC_DEF],
            )
        )
        assert resp.has_tool_calls
        assert len(resp.tool_calls) == 1
        tc = resp.tool_calls[0]
        assert tc.name == "calculator"
        assert tc.arguments == {"expression": "5 * 6"}
        assert tc.id == "call_xyz"
        assert resp.finish_reason is FinishReason.TOOL_CALLS
    finally:
        await client.close()


async def test_openai_parses_multiple_tool_calls() -> None:
    resp_data = {
        "id": "chatcmpl-1",
        "model": "cloud-coder:32b",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {
                                "name": "calculator",
                                "arguments": '{"expression": "1+1"}',
                            },
                        },
                        {
                            "id": "c2",
                            "type": "function",
                            "function": {
                                "name": "calculator",
                                "arguments": '{"expression": "2+2"}',
                            },
                        },
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
    }
    client, _ = _make_openai_client(resp_data)
    try:
        resp = await client.generate(
            LLMRequest(
                model="cloud-coder:32b",
                messages=[LLMMessage(role=MessageRole.USER, content="calc two things")],
                tools=[CALC_DEF],
            )
        )
        assert len(resp.tool_calls) == 2
        assert {tc.id for tc in resp.tool_calls} == {"c1", "c2"}
    finally:
        await client.close()


async def test_openai_arguments_as_dict_are_accepted() -> None:
    """Some providers return arguments as a dict instead of a JSON string."""
    resp_data = {
        "id": "chatcmpl-1",
        "model": "cloud-coder:32b",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {
                                "name": "calculator",
                                "arguments": {"expression": "8 / 2"},
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }
    client, _ = _make_openai_client(resp_data)
    try:
        resp = await client.generate(
            LLMRequest(
                model="cloud-coder:32b",
                messages=[LLMMessage(role=MessageRole.USER, content="calc")],
                tools=[CALC_DEF],
            )
        )
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].arguments == {"expression": "8 / 2"}
    finally:
        await client.close()


async def test_openai_malformed_tool_calls_are_skipped() -> None:
    resp_data = {
        "id": "chatcmpl-1",
        "model": "cloud-coder:32b",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "hmm",
                    "tool_calls": [
                        {"id": "c1", "type": "function", "function": {}},  # no name
                        {
                            "id": "c2",
                            "type": "function",
                            "function": {"name": "calculator", "arguments": "not-json"},
                        },
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }
    client, _ = _make_openai_client(resp_data)
    try:
        resp = await client.generate(
            LLMRequest(
                model="cloud-coder:32b",
                messages=[LLMMessage(role=MessageRole.USER, content="calc")],
                tools=[CALC_DEF],
            )
        )
        valid = [tc for tc in resp.tool_calls if tc.name]
        assert len(valid) == 1
        assert valid[0].arguments == {}
    finally:
        await client.close()


async def test_openai_no_tools_omits_tools_field() -> None:
    resp_data = {
        "id": "chatcmpl-1",
        "model": "cloud-coder:32b",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hi"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }
    client, requests = _make_openai_client(resp_data)
    try:
        await client.generate(
            LLMRequest(
                model="cloud-coder:32b",
                messages=[LLMMessage(role=MessageRole.USER, content="Hi")],
            )
        )
        payload = json.loads(requests[-1].content)
        assert "tools" not in payload
    finally:
        await client.close()


async def test_openai_assistant_tool_call_message_serialized() -> None:
    """Assistant messages with tool_calls are serialized in OpenAI format
    (arguments as a JSON string)."""
    resp_data = _openai_tool_call_response()
    client, requests = _make_openai_client(resp_data)
    try:
        tool_calls = [LLMToolCall(id="call_1", name="calculator", arguments={"expression": "2+2"})]
        await client.generate(
            LLMRequest(
                model="cloud-coder:32b",
                messages=[
                    LLMMessage(role=MessageRole.USER, content="What is 2+2?"),
                    LLMMessage(role=MessageRole.ASSISTANT, content="", tool_calls=tool_calls),
                ],
                tools=[CALC_DEF],
            )
        )
        payload = json.loads(requests[-1].content)
        assistant_msg = payload["messages"][1]
        assert assistant_msg["role"] == "assistant"
        assert "tool_calls" in assistant_msg
        # OpenAI serializes arguments as a JSON string.
        args = assistant_msg["tool_calls"][0]["function"]["arguments"]
        assert isinstance(args, str)
        assert json.loads(args) == {"expression": "2+2"}
    finally:
        await client.close()


async def test_openai_tool_result_message_serialized() -> None:
    resp_data = _openai_tool_call_response()
    client, requests = _make_openai_client(resp_data)
    try:
        await client.generate(
            LLMRequest(
                model="cloud-coder:32b",
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
