"""LLM abstraction layer (Phase 2).

Provider-independent LLM client, model/provider registries, factory, and a
deterministic local-first model router. JAR-63 is local-first: Ollama is the
primary provider and no cloud API key is required.

Agents and the planner depend on :class:`LLMClient`, never on a provider SDK.

::

    Agent / Planner
        ↓
    LLMClient (interface)
        ↓
    ModelRouter → ProviderFactory → Ollama / OpenAI-compatible / ...
"""

from __future__ import annotations

from app.llm.client import HealthStatus, LLMClient, ProviderHealth
from app.llm.errors import (
    InvalidLLMResponseError,
    InvalidStructuredOutputError,
    LLMAuthenticationError,
    LLMError,
    LLMPolicyError,
    LLMRateLimitError,
    LLMTimeoutError,
    ModelNotFoundError,
    ModelRoutingError,
    ModelUnavailableError,
    ProviderUnavailableError,
)
from app.llm.factory import ProviderFactory, UnknownProviderError
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
from app.llm.plan_schema import LLMPlan, LLMPlanStep
from app.llm.planner import LLMPlanner, PlannerFallbackReason
from app.llm.providers.ollama import OLLAMA_PROVIDER_ID, OllamaClient
from app.llm.providers.openai_compatible import (
    OPENAI_COMPATIBLE_PROVIDER_ID,
    OpenAICompatibleClient,
)
from app.llm.registry import (
    ModelAlreadyRegisteredError,
    ModelNotFoundError_,
    ModelRegistry,
    ProviderAlreadyRegisteredError,
    ProviderNotFoundError,
    ProviderRegistry,
)
from app.llm.router import (
    AlwaysOfflineNetworkChecker,
    ModelRouter,
    ModelSelection,
    NetworkChecker,
    RoutingPolicy,
    RoutingRequest,
)
from app.llm.tool_conversion import (
    build_assistant_tool_call_message,
    build_tool_result_message,
    tool_info_to_definition,
    tool_to_definition,
    tools_to_definitions,
)
from app.llm.verifier import LLMVerifier

__all__ = [
    "AlwaysOfflineNetworkChecker",
    "FinishReason",
    "HealthStatus",
    "InvalidLLMResponseError",
    "InvalidStructuredOutputError",
    "LLMAuthenticationError",
    "LLMClient",
    "LLMError",
    "LLMMessage",
    "LLMPlan",
    "LLMPlanStep",
    "LLMPlanner",
    "LLMPolicyError",
    "LLMRateLimitError",
    "LLMRequest",
    "LLMResponse",
    "LLMTimeoutError",
    "LLMToolCall",
    "LLMToolDefinition",
    "LLMUsage",
    "LLMVerifier",
    "MessageRole",
    "ModelAlreadyRegisteredError",
    "ModelCapability",
    "ModelDefinition",
    "ModelNotFoundError",
    "ModelNotFoundError_",
    "ModelRegistry",
    "ModelRouter",
    "ModelRoutingError",
    "ModelSelection",
    "ModelUnavailableError",
    "NetworkChecker",
    "OLLAMA_PROVIDER_ID",
    "OPENAI_COMPATIBLE_PROVIDER_ID",
    "OllamaClient",
    "OpenAICompatibleClient",
    "PlannerFallbackReason",
    "PrivacyLevel",
    "ProviderAlreadyRegisteredError",
    "ProviderFactory",
    "ProviderHealth",
    "ProviderNotFoundError",
    "ProviderRegistry",
    "ProviderUnavailableError",
    "RoutingPolicy",
    "RoutingRequest",
    "StructuredOutputSpec",
    "UnknownProviderError",
    "build_assistant_tool_call_message",
    "build_tool_result_message",
    "tool_info_to_definition",
    "tool_to_definition",
    "tools_to_definitions",
]
