"""Tests for the Ollama provider (mocked HTTP, fully offline)."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel as PydanticBaseModel

from app.events import EventType
from app.llm.client import HealthStatus
from app.llm.errors import InvalidStructuredOutputError, LLMTimeoutError, ModelNotFoundError
from app.llm.models import LLMMessage, LLMRequest, MessageRole, StructuredOutputSpec
from tests.llm_helpers import OllamaMockTransport


async def test_ollama_generate_returns_normalized_response() -> None:
    mock = OllamaMockTransport()
    client = mock.make_client()
    try:
        resp = await client.generate(
            LLMRequest(
                model="qwen2.5-coder:7b",
                messages=[LLMMessage(role=MessageRole.USER, content="Hi")],
            )
        )
        assert resp.content == "Hello!"
        assert resp.provider == "ollama"
        assert resp.model == "qwen2.5-coder:7b"
        assert resp.usage.prompt_tokens == 10
        assert resp.usage.completion_tokens == 3
        assert resp.usage.total_tokens == 13
    finally:
        await client.close()


async def test_ollama_health_available() -> None:
    mock = OllamaMockTransport(installed_models=["qwen2.5-coder:7b", "llama3.2:3b"])
    client = mock.make_client()
    try:
        health = await client.check_health()
        assert health.status == HealthStatus.AVAILABLE
        assert "2 model" in (health.detail or "")
    finally:
        await client.close()


async def test_ollama_health_model_available() -> None:
    mock = OllamaMockTransport(installed_models=["qwen2.5-coder:7b"])
    client = mock.make_client()
    try:
        health = await client.check_health("qwen2.5-coder:7b")
        assert health.available
        assert health.model == "qwen2.5-coder:7b"
    finally:
        await client.close()


async def test_ollama_health_model_not_found() -> None:
    mock = OllamaMockTransport(installed_models=["qwen2.5-coder:7b"])
    client = mock.make_client()
    try:
        health = await client.check_health("missing-model:7b")
        assert health.status == HealthStatus.MODEL_NOT_FOUND
        assert "ollama pull missing-model:7b" in (health.detail or "")
    finally:
        await client.close()


async def test_ollama_health_unreachable() -> None:
    mock = OllamaMockTransport(raise_connect=True)
    client = mock.make_client()
    try:
        health = await client.check_health()
        assert health.status == HealthStatus.UNAVAILABLE
    finally:
        await client.close()


async def test_ollama_health_timeout() -> None:
    mock = OllamaMockTransport(raise_timeout=True)
    client = mock.make_client()
    try:
        health = await client.check_health()
        assert health.status == HealthStatus.TIMEOUT
    finally:
        await client.close()


async def test_ollama_model_not_found_on_generate() -> None:
    mock = OllamaMockTransport(chat_status=404)
    client = mock.make_client()
    try:
        with pytest.raises(ModelNotFoundError, match="ollama pull"):
            await client.generate(
                LLMRequest(
                    model="missing:7b",
                    messages=[LLMMessage(role=MessageRole.USER, content="Hi")],
                )
            )
    finally:
        await client.close()


async def test_ollama_timeout_on_generate() -> None:
    mock = OllamaMockTransport(raise_timeout=True)
    client = mock.make_client()
    try:
        with pytest.raises(LLMTimeoutError):
            await client.generate(
                LLMRequest(
                    model="qwen2.5-coder:7b",
                    messages=[LLMMessage(role=MessageRole.USER, content="Hi")],
                )
            )
    finally:
        await client.close()


async def test_ollama_generate_structured_with_response_model() -> None:
    class Answer(PydanticBaseModel):
        value: int
        label: str

    chat_response = {
        "model": "qwen2.5-coder:7b",
        "message": {"role": "assistant", "content": json.dumps({"value": 42, "label": "answer"})},
        "done": True,
        "eval_count": 5,
    }
    mock = OllamaMockTransport(chat_response=chat_response)
    client = mock.make_client()
    try:
        spec = StructuredOutputSpec(name="answer", response_model=Answer)
        resp = await client.generate_structured(
            LLMRequest(
                model="qwen2.5-coder:7b",
                messages=[LLMMessage(role=MessageRole.USER, content="What is 42?")],
                structured_output=spec,
            )
        )
        parsed = resp.metadata["parsed"]
        assert parsed["value"] == 42
        assert parsed["label"] == "answer"
    finally:
        await client.close()


async def test_ollama_generate_structured_invalid_json() -> None:
    chat_response = {
        "model": "qwen2.5-coder:7b",
        "message": {"role": "assistant", "content": "not json"},
        "done": True,
    }
    mock = OllamaMockTransport(chat_response=chat_response)
    client = mock.make_client()
    try:
        spec = StructuredOutputSpec(name="answer", json_schema={"type": "object"})
        with pytest.raises(InvalidStructuredOutputError, match="not valid JSON"):
            await client.generate_structured(
                LLMRequest(
                    model="qwen2.5-coder:7b",
                    messages=[LLMMessage(role=MessageRole.USER, content="Hi")],
                    structured_output=spec,
                )
            )
    finally:
        await client.close()


async def test_ollama_generate_structured_schema_validation_failure() -> None:
    class Answer(PydanticBaseModel):
        value: int

    chat_response = {
        "model": "qwen2.5-coder:7b",
        "message": {"role": "assistant", "content": json.dumps({"value": "not-an-int"})},
        "done": True,
    }
    mock = OllamaMockTransport(chat_response=chat_response)
    client = mock.make_client()
    try:
        spec = StructuredOutputSpec(name="answer", response_model=Answer)
        with pytest.raises(InvalidStructuredOutputError, match="schema validation"):
            await client.generate_structured(
                LLMRequest(
                    model="qwen2.5-coder:7b",
                    messages=[LLMMessage(role=MessageRole.USER, content="Hi")],
                    structured_output=spec,
                )
            )
    finally:
        await client.close()


async def test_ollama_generate_structured_requires_spec() -> None:
    mock = OllamaMockTransport()
    client = mock.make_client()
    try:
        with pytest.raises(InvalidStructuredOutputError, match="No structured"):
            await client.generate_structured(
                LLMRequest(
                    model="qwen2.5-coder:7b",
                    messages=[LLMMessage(role=MessageRole.USER, content="Hi")],
                )
            )
    finally:
        await client.close()


async def test_ollama_discover_models() -> None:
    mock = OllamaMockTransport(installed_models=["qwen2.5-coder:7b", "llama3.2:3b"])
    client = mock.make_client()
    try:
        models = await client.discover_models()
        assert set(models) == {"qwen2.5-coder:7b", "llama3.2:3b"}
    finally:
        await client.close()


async def test_ollama_get_model_info() -> None:
    mock = OllamaMockTransport()
    client = mock.make_client()
    try:
        info = await client.get_model_info("qwen2.5-coder:7b")
        assert info is not None
        assert info.provider == "ollama"
        assert info.local is True
        assert info.context_window == 32768
    finally:
        await client.close()


async def test_ollama_publishes_events(captured_events) -> None:
    bus, events = captured_events
    mock = OllamaMockTransport()
    client = mock.make_client(event_bus=bus)
    try:
        await client.generate(
            LLMRequest(
                model="qwen2.5-coder:7b",
                messages=[LLMMessage(role=MessageRole.USER, content="Hi")],
            )
        )
        types = [e.event_type for e in events]
        assert EventType.LLM_REQUEST_STARTED in types
        assert EventType.LLM_REQUEST_COMPLETED in types
        # No prompt content in events.
        for e in events:
            assert "Hi" not in json.dumps(e.payload)
    finally:
        await client.close()


async def test_ollama_publishes_failure_event(captured_events) -> None:
    bus, events = captured_events
    mock = OllamaMockTransport(raise_timeout=True)
    client = mock.make_client(event_bus=bus)
    try:
        with pytest.raises(LLMTimeoutError):
            await client.generate(
                LLMRequest(
                    model="qwen2.5-coder:7b",
                    messages=[LLMMessage(role=MessageRole.USER, content="Hi")],
                )
            )
        types = [e.event_type for e in events]
        assert EventType.LLM_REQUEST_STARTED in types
        assert EventType.LLM_REQUEST_FAILED in types
    finally:
        await client.close()
