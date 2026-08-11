"""Strongly typed LLM models.

These are the provider-independent data structures that flow through the
LLM abstraction layer. Provider-specific raw responses are normalized into
these models at the provider boundary and never leak into the rest of the
application.

Models:
* :class:`MessageRole`, :class:`LLMMessage`
* :class:`LLMToolDefinition`, :class:`LLMToolCall`
* :class:`LLMUsage`
* :class:`PrivacyLevel`
* :class:`FinishReason`
* :class:`LLMResponse`
* :class:`StructuredOutputSpec`
* :class:`ModelCapability`, :class:`ModelDefinition`
* :class:`LLMRequest`
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.core.identifiers import generate_id, utc_now

__all__ = [
    "MessageRole",
    "LLMMessage",
    "LLMToolDefinition",
    "LLMToolCall",
    "LLMUsage",
    "PrivacyLevel",
    "FinishReason",
    "LLMResponse",
    "StructuredOutputSpec",
    "ModelCapability",
    "ModelDefinition",
    "LLMRequest",
]


class MessageRole(StrEnum):
    """Roles a message may take in a chat completion."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class PrivacyLevel(StrEnum):
    """Privacy classification of an LLM request.

    ``PRIVATE`` and ``SENSITIVE`` requests prefer local models and are not
    sent to cloud providers unless explicitly allowed by policy.
    """

    PUBLIC = "public"
    INTERNAL = "internal"
    PRIVATE = "private"
    SENSITIVE = "sensitive"

    @property
    def is_cloud_restricted(self) -> bool:
        """Whether cloud providers are disallowed by default for this level."""
        return self in (PrivacyLevel.PRIVATE, PrivacyLevel.SENSITIVE)


class FinishReason(StrEnum):
    """Why the model stopped generating."""

    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    UNKNOWN = "unknown"


class ModelCapability(StrEnum):
    """Extensible capability tags a model may declare.

    New capabilities can be added in later phases without breaking existing
    model definitions.
    """

    CHAT = "chat"
    CODING = "coding"
    REASONING = "reasoning"
    VISION = "vision"
    TOOL_CALLING = "tool_calling"
    STRUCTURED_OUTPUT = "structured_output"
    LONG_CONTEXT = "long_context"


class LLMMessage(BaseModel):
    """A single chat message."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: MessageRole
    content: str
    name: str | None = None
    # For assistant messages that request tool execution.
    tool_calls: list[LLMToolCall] = Field(default_factory=list)
    # For tool-role messages: the id of the tool call this responds to.
    tool_call_id: str | None = None


class LLMToolCall(BaseModel):
    """A tool invocation requested by the assistant."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: generate_id("call"))
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class LLMToolDefinition(BaseModel):
    """Definition of a tool the model may call.

    ``parameters`` is a JSON Schema object describing the tool's arguments.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class LLMUsage(BaseModel):
    """Token usage for a request (best-effort; not all providers report it)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    @property
    def has_data(self) -> bool:
        return any(
            v is not None for v in (self.prompt_tokens, self.completion_tokens, self.total_tokens)
        )


class LLMResponse(BaseModel):
    """Normalized, provider-independent response.

    Provider raw response objects are never stored here.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    response_id: str = Field(default_factory=lambda: generate_id("resp"))
    content: str = ""
    model: str
    provider: str
    usage: LLMUsage = Field(default_factory=LLMUsage)
    finish_reason: FinishReason = FinishReason.UNKNOWN
    tool_calls: list[LLMToolCall] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class StructuredOutputSpec(BaseModel):
    """Specification for structured (JSON-schema-conformant) output.

    Either a JSON Schema (``schema``) or a Pydantic model class
    (``response_model``) may be provided. When ``response_model`` is given,
    its JSON Schema is derived at request time.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    name: str = "structured_output"
    json_schema: dict[str, Any] | None = None
    response_model: type[BaseModel] | None = None

    def to_json_schema(self) -> dict[str, Any]:
        """Return the JSON Schema describing the expected output."""
        if self.json_schema is not None:
            return self.json_schema
        if self.response_model is not None:
            return self.response_model.model_json_schema()
        raise ValueError("StructuredOutputSpec has neither json_schema nor response_model")


class ModelDefinition(BaseModel):
    """Static description of an available model.

    Capabilities are extensible (:class:`ModelCapability`). Provider/model
    independence is preserved: a model is identified by ``model_id`` and
    ``provider`` together.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str
    provider: str
    display_name: str = ""
    capabilities: set[ModelCapability] = Field(default_factory=set)
    context_window: int | None = None
    supports_tools: bool = False
    supports_structured_output: bool = False
    supports_vision: bool = False
    supports_reasoning: bool = False
    coding_score: float | None = None
    local: bool = False
    enabled: bool = True

    @property
    def display(self) -> str:
        return self.display_name or self.model_id

    def has_capability(self, capability: ModelCapability) -> bool:
        return capability in self.capabilities


class LLMRequest(BaseModel):
    """A provider-independent LLM request.

    No provider-specific parameters are exposed through the core interface.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    model: str
    messages: list[LLMMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    timeout: float | None = None
    tools: list[LLMToolDefinition] = Field(default_factory=list)
    structured_output: StructuredOutputSpec | None = None
    privacy: PrivacyLevel = PrivacyLevel.PUBLIC
    metadata: dict[str, Any] = Field(default_factory=dict)
    request_id: str = Field(default_factory=lambda: generate_id("llm_req"))
