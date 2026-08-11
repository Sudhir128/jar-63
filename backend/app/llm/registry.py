"""Model and provider registries.

Both registries are provider-independent: they store :class:`ModelDefinition`
metadata and :class:`LLMClient` instances respectively, never provider SDKs.
Agents look up models/providers through these registries (or via the router),
never directly.
"""

from __future__ import annotations

from app.core.exceptions import JARError
from app.core.logging import get_logger
from app.llm.client import HealthStatus, LLMClient, ProviderHealth
from app.llm.models import ModelCapability, ModelDefinition

logger = get_logger("llm.registry")

__all__ = [
    "ModelNotFoundError_",
    "ModelAlreadyRegisteredError",
    "ProviderNotFoundError",
    "ProviderAlreadyRegisteredError",
    "ModelRegistry",
    "ProviderRegistry",
]


class ModelNotFoundError_(JARError):
    """Raised when a requested model is not in the registry."""


class ModelAlreadyRegisteredError(JARError):
    """Raised when registering a model whose id+provider already exists."""


class ProviderNotFoundError(JARError):
    """Raised when a requested provider is not registered."""


class ProviderAlreadyRegisteredError(JARError):
    """Raised when registering a provider whose id already exists."""


class ModelRegistry:
    """In-process registry of model definitions.

    Keys are ``(provider, model_id)`` pairs so the same model id may exist
    across providers. The registry depends only on :class:`ModelDefinition`.
    """

    def __init__(self) -> None:
        self._models: dict[tuple[str, str], ModelDefinition] = {}

    def register(self, model: ModelDefinition) -> ModelDefinition:
        key = (model.provider, model.model_id)
        if key in self._models:
            raise ModelAlreadyRegisteredError(f"{model.provider}/{model.model_id}")
        self._models[key] = model
        logger.bind(event="model.registered", provider=model.provider, model=model.model_id).info(
            "Registered model '{}'", model.display
        )
        return model

    def unregister(self, provider: str, model_id: str) -> ModelDefinition:
        key = (provider, model_id)
        model = self._models.pop(key, None)
        if model is None:
            raise ModelNotFoundError_(f"{provider}/{model_id}")
        return model

    def get(self, provider: str, model_id: str) -> ModelDefinition:
        model = self._models.get((provider, model_id))
        if model is None:
            raise ModelNotFoundError_(f"{provider}/{model_id}")
        return model

    def list(self) -> list[ModelDefinition]:
        return list(self._models.values())

    def list_enabled(self) -> list[ModelDefinition]:
        return [m for m in self._models.values() if m.enabled]

    def exists(self, provider: str, model_id: str) -> bool:
        return (provider, model_id) in self._models

    def find_by_capability(self, capability: ModelCapability) -> list[ModelDefinition]:
        return [m for m in self._models.values() if capability in m.capabilities]

    def find_enabled_by_capability(self, capability: ModelCapability) -> list[ModelDefinition]:
        return [m for m in self._models.values() if m.enabled and capability in m.capabilities]

    def enable(self, provider: str, model_id: str) -> ModelDefinition:
        return self._set_enabled(provider, model_id, True)

    def disable(self, provider: str, model_id: str) -> ModelDefinition:
        return self._set_enabled(provider, model_id, False)

    def _set_enabled(self, provider: str, model_id: str, enabled: bool) -> ModelDefinition:
        key = (provider, model_id)
        model = self._models.get(key)
        if model is None:
            raise ModelNotFoundError_(f"{provider}/{model_id}")
        updated = model.model_copy(update={"enabled": enabled})
        self._models[key] = updated
        return updated

    def __len__(self) -> int:
        return len(self._models)


class ProviderRegistry:
    """In-process registry of LLM clients (providers).

    Depends only on :class:`LLMClient`, never on provider SDKs.
    """

    def __init__(self) -> None:
        self._providers: dict[str, LLMClient] = {}

    def register(self, provider_id: str, client: LLMClient) -> LLMClient:
        if provider_id in self._providers:
            raise ProviderAlreadyRegisteredError(provider_id)
        self._providers[provider_id] = client
        logger.bind(event="provider.registered", provider=provider_id).info(
            "Registered LLM provider '{}'", provider_id
        )
        return client

    def unregister(self, provider_id: str) -> LLMClient:
        client = self._providers.pop(provider_id, None)
        if client is None:
            raise ProviderNotFoundError(provider_id)
        return client

    def get(self, provider_id: str) -> LLMClient:
        client = self._providers.get(provider_id)
        if client is None:
            raise ProviderNotFoundError(provider_id)
        return client

    def list(self) -> list[str]:
        return list(self._providers.keys())

    def exists(self, provider_id: str) -> bool:
        return provider_id in self._providers

    async def health_check(self, provider_id: str, model: str | None = None) -> ProviderHealth:
        """Check the health of a registered provider (and optional model)."""
        client = self._providers.get(provider_id)
        if client is None:
            return ProviderHealth(
                status=HealthStatus.UNAVAILABLE,
                provider=provider_id,
                detail="provider not registered",
            )
        try:
            return await client.check_health(model)
        except Exception as exc:  # noqa: BLE001 - health checks must never raise
            logger.bind(
                event="provider.health.error", provider=provider_id, error=type(exc).__name__
            ).warning("Health check failed: {}", str(exc))
            return ProviderHealth(
                status=HealthStatus.UNAVAILABLE,
                provider=provider_id,
                model=model,
                detail=str(exc),
            )

    def __len__(self) -> int:
        return len(self._providers)
