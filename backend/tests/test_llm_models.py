"""Tests for LLM typed models (messages, requests, responses, usage, tools)."""

from __future__ import annotations

import pytest
from pydantic import BaseModel as PydanticBaseModel
from pydantic import ValidationError

from app.llm.models import (
    FinishReason,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMToolCall,
    LLMToolDefinition,
    LLMUsage,
    MessageRole,
    ModelCapability,
    ModelDefinition,
    PrivacyLevel,
    StructuredOutputSpec,
)


def test_message_roles() -> None:
    assert MessageRole.SYSTEM.value == "system"
    assert MessageRole.USER.value == "user"
    assert MessageRole.ASSISTANT.value == "assistant"
    assert MessageRole.TOOL.value == "tool"


def test_llm_message_basic() -> None:
    msg = LLMMessage(role=MessageRole.USER, content="Hello")
    assert msg.role is MessageRole.USER
    assert msg.content == "Hello"
    assert msg.tool_calls == []


def test_llm_message_is_frozen() -> None:
    msg = LLMMessage(role=MessageRole.USER, content="Hello")
    with pytest.raises(ValidationError):
        msg.content = "mutated"  # type: ignore[misc]


def test_llm_message_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        LLMMessage(role=MessageRole.USER, content="Hi", bogus=True)  # type: ignore[call-arg]


def test_llm_tool_definition() -> None:
    tool = LLMToolDefinition(
        name="search", description="Search the web", parameters={"type": "object"}
    )
    assert tool.name == "search"
    assert tool.parameters == {"type": "object"}


def test_llm_tool_call() -> None:
    call = LLMToolCall(name="search", arguments={"q": "pydantic"})
    assert call.name == "search"
    assert call.arguments == {"q": "pydantic"}
    assert call.id.startswith("call_")


def test_llm_usage() -> None:
    usage = LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    assert usage.has_data is True
    assert LLMUsage().has_data is False


def test_llm_response() -> None:
    resp = LLMResponse(
        content="Hi",
        model="qwen2.5-coder:7b",
        provider="ollama",
        usage=LLMUsage(prompt_tokens=2, completion_tokens=1, total_tokens=3),
        finish_reason=FinishReason.STOP,
    )
    assert resp.content == "Hi"
    assert resp.provider == "ollama"
    assert resp.has_tool_calls is False
    assert resp.usage.total_tokens == 3


def test_privacy_levels() -> None:
    assert PrivacyLevel.PUBLIC.is_cloud_restricted is False
    assert PrivacyLevel.INTERNAL.is_cloud_restricted is False
    assert PrivacyLevel.PRIVATE.is_cloud_restricted is True
    assert PrivacyLevel.SENSITIVE.is_cloud_restricted is True


def test_structured_output_with_json_schema() -> None:
    spec = StructuredOutputSpec(
        name="answer", json_schema={"type": "object", "properties": {"value": {"type": "integer"}}}
    )
    schema = spec.to_json_schema()
    assert schema["type"] == "object"
    assert "value" in schema["properties"]


def test_structured_output_with_response_model() -> None:
    class Answer(PydanticBaseModel):
        value: int

    spec = StructuredOutputSpec(name="answer", response_model=Answer)
    schema = spec.to_json_schema()
    assert "properties" in schema
    assert "value" in schema["properties"]


def test_structured_output_requires_schema_or_model() -> None:
    spec = StructuredOutputSpec(name="empty")
    with pytest.raises(ValueError, match="neither"):
        spec.to_json_schema()


def test_model_definition() -> None:
    model = ModelDefinition(
        model_id="qwen2.5-coder:7b",
        provider="ollama",
        display_name="Qwen Coder 7B",
        capabilities={ModelCapability.CHAT, ModelCapability.CODING},
        context_window=32768,
        supports_structured_output=True,
        local=True,
        enabled=True,
    )
    assert model.has_capability(ModelCapability.CODING)
    assert not model.has_capability(ModelCapability.VISION)
    assert model.display == "Qwen Coder 7B"
    assert ModelDefinition(model_id="x", provider="ollama").display == "x"


def test_llm_request() -> None:
    req = LLMRequest(
        model="qwen2.5-coder:7b",
        messages=[LLMMessage(role=MessageRole.USER, content="Hi")],
        temperature=0.5,
        max_tokens=100,
        privacy=PrivacyLevel.PRIVATE,
    )
    assert req.model == "qwen2.5-coder:7b"
    assert req.temperature == 0.5
    assert req.privacy is PrivacyLevel.PRIVATE
    assert req.request_id.startswith("llm_req_")


def test_model_capability_extensible() -> None:
    # The enum is extensible: existing capabilities are present.
    assert ModelCapability.CHAT.value == "chat"
    assert ModelCapability.CODING.value == "coding"
    assert ModelCapability.TOOL_CALLING.value == "tool_calling"
