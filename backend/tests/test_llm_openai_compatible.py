"""Tests for the generic OpenAI-compatible provider (mocked, offline)."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel as PydanticBaseModel

from app.llm.errors import InvalidStructuredOutputError, LLMAuthenticationError
from app.llm.models import LLMMessage, LLMRequest, MessageRole, StructuredOutputSpec
from tests.llm_helpers import OpenAICompatMockTransport


async def test_openai_compat_generate() -> None:
    mock = OpenAICompatMockTransport()
    client = mock.make_client()
    try:
        resp = await client.generate(
            LLMRequest(
                model="cloud-coder:32b",
                messages=[LLMMessage(role=MessageRole.USER, content="Hi")],
            )
        )
        assert resp.content == "Hi from cloud"
        assert resp.provider == "openai_compatible"
        assert resp.usage.total_tokens == 12
    finally:
        await client.close()


async def test_openai_compat_requires_base_url_and_model() -> None:
    from app.llm.providers.openai_compatible import OpenAICompatibleClient

    with pytest.raises(ValueError, match="base_url"):
        OpenAICompatibleClient(base_url="", model="m")
    with pytest.raises(ValueError, match="model"):
        OpenAICompatibleClient(base_url="http://x", model="")


async def test_openai_compat_api_key_optional() -> None:
    mock = OpenAICompatMockTransport()
    client = mock.make_client(api_key=None)
    try:
        resp = await client.generate(
            LLMRequest(
                model="cloud-coder:32b",
                messages=[LLMMessage(role=MessageRole.USER, content="Hi")],
            )
        )
        assert resp.content == "Hi from cloud"
        # No Authorization header when no key.
        assert all(h is None for h in mock.auth_headers)
    finally:
        await client.close()


async def test_openai_compat_api_key_sent_as_bearer() -> None:
    mock = OpenAICompatMockTransport()
    client = mock.make_client(api_key="sk-test-key")
    try:
        await client.generate(
            LLMRequest(
                model="cloud-coder:32b",
                messages=[LLMMessage(role=MessageRole.USER, content="Hi")],
            )
        )
        auth = mock.auth_headers[-1]
        assert auth == "Bearer sk-test-key"
    finally:
        await client.close()


async def test_openai_compat_auth_failure() -> None:
    mock = OpenAICompatMockTransport(status=401)
    client = mock.make_client(api_key="sk-bad")
    try:
        with pytest.raises(LLMAuthenticationError):
            await client.generate(
                LLMRequest(
                    model="cloud-coder:32b",
                    messages=[LLMMessage(role=MessageRole.USER, content="Hi")],
                )
            )
    finally:
        await client.close()


async def test_openai_compat_structured_output() -> None:
    class Answer(PydanticBaseModel):
        value: int

    completion = {
        "id": "chatcmpl-1",
        "model": "cloud-coder:32b",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": json.dumps({"value": 7})},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }
    mock = OpenAICompatMockTransport(completion_response=completion)
    client = mock.make_client()
    try:
        spec = StructuredOutputSpec(name="answer", response_model=Answer)
        resp = await client.generate_structured(
            LLMRequest(
                model="cloud-coder:32b",
                messages=[LLMMessage(role=MessageRole.USER, content="What is 7?")],
                structured_output=spec,
            )
        )
        assert resp.metadata["parsed"]["value"] == 7
    finally:
        await client.close()


async def test_openai_compat_structured_invalid_json() -> None:
    completion = {
        "id": "chatcmpl-1",
        "model": "cloud-coder:32b",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "not json"},
                "finish_reason": "stop",
            }
        ],
        "usage": {},
    }
    mock = OpenAICompatMockTransport(completion_response=completion)
    client = mock.make_client()
    try:
        spec = StructuredOutputSpec(name="answer", json_schema={"type": "object"})
        with pytest.raises(InvalidStructuredOutputError, match="not valid JSON"):
            await client.generate_structured(
                LLMRequest(
                    model="cloud-coder:32b",
                    messages=[LLMMessage(role=MessageRole.USER, content="Hi")],
                    structured_output=spec,
                )
            )
    finally:
        await client.close()


async def test_openai_compat_health_unreachable() -> None:
    mock = OpenAICompatMockTransport(raise_connect=True)
    client = mock.make_client()
    try:
        health = await client.check_health()
        from app.llm.client import HealthStatus

        assert health.status == HealthStatus.UNAVAILABLE
    finally:
        await client.close()
