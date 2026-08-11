"""Provider-independent LLM client interface.

Agents and the planner depend on :class:`LLMClient`, never on a specific
provider SDK. The client is async-first. Provider implementations live in
:mod:`app.llm.providers`.

::

    Agent / Planner
        ↓
    LLMClient (interface)
        ↓
    Model Router → Selected Provider (Ollama / OpenAI-compatible / ...)
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from app.llm.models import LLMRequest, LLMResponse, ModelDefinition

__all__ = ["LLMClient", "ProviderHealth", "HealthStatus"]


class HealthStatus(StrEnum):
    """Conceptual health states for a provider/model.

    * ``AVAILABLE``        — provider reachable, model usable.
    * ``UNAVAILABLE``      — provider not reachable.
    * ``MODEL_NOT_FOUND``  — provider reachable but model not installed/known.
    * ``TIMEOUT``          — health check timed out.
    * ``INVALID_RESPONSE`` — provider returned an unusable response.
    """

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    MODEL_NOT_FOUND = "model_not_found"
    TIMEOUT = "timeout"
    INVALID_RESPONSE = "invalid_response"


@dataclass(frozen=True)
class ProviderHealth:
    """Result of a provider/model health check."""

    status: str
    provider: str
    model: str | None = None
    detail: str | None = None

    @property
    def available(self) -> bool:
        return self.status == HealthStatus.AVAILABLE


class LLMClient(abc.ABC):
    """Abstract, provider-independent LLM client contract.

    Implementations must be safe to construct without network access —
    connection is established lazily on the first request or health check.
    """

    @property
    @abc.abstractmethod
    def provider_name(self) -> str:
        """Return the provider identifier (e.g. ``ollama``)."""

    @abc.abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a chat completion and return a normalized response."""

    @abc.abstractmethod
    async def generate_structured(
        self, request: LLMRequest, spec: Any | None = None
    ) -> LLMResponse:
        """Generate a completion constrained to a structured output spec.

        ``spec`` overrides ``request.structured_output`` when provided. The
        returned response carries parsed JSON in ``metadata["parsed"]`` when
        a response model/schema is available.
        """

    @abc.abstractmethod
    async def check_health(self, model: str | None = None) -> ProviderHealth:
        """Check provider reachability and optional model availability."""

    @abc.abstractmethod
    async def get_model_info(self, model: str) -> ModelDefinition | None:
        """Return model info if the model is known/installed, else ``None``."""

    @abc.abstractmethod
    async def close(self) -> None:
        """Release any underlying resources (HTTP clients, etc.)."""

    async def __aenter__(self) -> LLMClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
