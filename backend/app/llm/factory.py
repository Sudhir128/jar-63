"""LLM provider factory.

Given a provider id or a :class:`ModelDefinition`, build the correct
:class:`LLMClient` from configuration. The factory uses dependency injection
(optional event bus / HTTP client) and never hardcodes credentials.

::

    ModelDefinition / provider config
        ↓
    ProviderFactory
        ↓
    OllamaClient  |  OpenAICompatibleClient
"""

from __future__ import annotations

import httpx

from app.config import LLMSettings
from app.events import EventBus
from app.llm.client import LLMClient
from app.llm.models import ModelDefinition
from app.llm.providers.ollama import OLLAMA_PROVIDER_ID, OllamaClient
from app.llm.providers.openai_compatible import (
    OPENAI_COMPATIBLE_PROVIDER_ID,
    OpenAICompatibleClient,
)

__all__ = ["ProviderFactory", "UnknownProviderError"]


class UnknownProviderError(ValueError):
    """Raised when a provider id has no known implementation."""


class ProviderFactory:
    """Builds :class:`LLMClient` instances from settings.

    The factory is the single place that knows how to construct concrete
    providers. Callers depend on :class:`LLMClient`, never on the concrete
    classes.
    """

    def __init__(
        self,
        settings: LLMSettings,
        *,
        event_bus: EventBus | None = None,
    ) -> None:
        self._settings = settings
        self._event_bus = event_bus

    def create(
        self,
        provider_id: str,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> LLMClient:
        """Build a client for ``provider_id`` from settings."""
        if provider_id == OLLAMA_PROVIDER_ID:
            return OllamaClient(
                base_url=self._settings.ollama_base_url,
                timeout=self._settings.request_timeout,
                verbose_logging=self._settings.verbose_logging,
                http_client=http_client,
                event_bus=self._event_bus,
            )
        if provider_id == OPENAI_COMPATIBLE_PROVIDER_ID:
            if not self._settings.openai_compatible_configured:
                raise UnknownProviderError(
                    "OpenAI-compatible provider is not configured "
                    "(set OPENAI_COMPATIBLE_ENABLED, BASE_URL, and MODEL)."
                )
            api_key = (
                self._settings.openai_compatible_api_key.get_secret_value()
                if self._settings.openai_compatible_api_key
                else None
            )
            return OpenAICompatibleClient(
                base_url=self._settings.openai_compatible_base_url,
                model=self._settings.openai_compatible_model,
                api_key=api_key,
                timeout=self._settings.request_timeout,
                verbose_logging=self._settings.verbose_logging,
                http_client=http_client,
                event_bus=self._event_bus,
            )
        raise UnknownProviderError(f"Unknown LLM provider: {provider_id}")

    def create_for_model(
        self,
        model: ModelDefinition,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> LLMClient:
        """Build a client suitable for the given model definition."""
        return self.create(model.provider, http_client=http_client)
