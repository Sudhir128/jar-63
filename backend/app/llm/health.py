"""LLM runtime health checker and model discovery (Phase 5).

At startup (or on demand) this module:

1. Checks whether the Ollama server is reachable (``GET /api/tags``).
2. Discovers installed models and their capabilities.
3. Syncs discovered models into the :class:`ModelRegistry`, marking the
   configured default model as available (or ``MODEL_NOT_FOUND``).
4. Records a :class:`LLMHealthSnapshot` that the health/status API can expose.

The check **never raises** and **never fails application startup**. When
Ollama is unavailable, the snapshot records ``LOCAL_LLM_UNAVAILABLE`` and the
runtime continues with the deterministic fallback (and optional cloud).

This module does not download, install, or modify models. It only reads
Ollama's HTTP API. It never invokes ``ollama run`` or any CLI subprocess.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any

from app.config import LLMSettings
from app.core.logging import get_logger
from app.events import Event, EventBus, EventType
from app.llm.client import HealthStatus, LLMClient, ProviderHealth
from app.llm.models import ModelCapability, ModelDefinition
from app.llm.providers.ollama import OLLAMA_PROVIDER_ID
from app.llm.registry import ModelRegistry, ProviderRegistry

logger = get_logger("llm.health")

__all__ = [
    "LLMHealthSnapshot",
    "LLMHealthChecker",
    "DEFAULT_HEALTH_TIMEOUT",
]


DEFAULT_HEALTH_TIMEOUT = 10.0


def _capability_str(val: Any) -> str:
    return str(val).lower() if val is not None else ""


@dataclass
class LLMHealthSnapshot:
    """Immutable snapshot of LLM subsystem health at a point in time.

    Exposed via the health/status API. Contains **no secrets** — only
    provider/model/capability/availability metadata.
    """

    provider: str
    status: str
    model: str | None = None
    model_available: bool = False
    installed_models: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    routing_policy: str = "local_first"
    fallback_available: bool = True
    cloud_enabled: bool = False
    detail: str | None = None
    checked_at: str = ""

    @property
    def available(self) -> bool:
        """True when the local LLM is usable for requests right now."""
        return self.status == HealthStatus.AVAILABLE.value and self.model_available

    @property
    def degraded(self) -> bool:
        """True when the provider is present but the model is unusable."""
        return self.status in (
            HealthStatus.MODEL_NOT_FOUND.value,
            HealthStatus.UNAVAILABLE.value,
            HealthStatus.TIMEOUT.value,
            HealthStatus.INVALID_RESPONSE.value,
        )

    def to_api_dict(self) -> dict[str, Any]:
        """Serialize for the REST API (no secrets)."""
        return {
            "status": "ok" if self.available else "degraded",
            "provider": self.provider,
            "model": self.model,
            "model_available": self.model_available,
            "health_status": self.status,
            "installed_models": list(self.installed_models),
            "capabilities": list(self.capabilities),
            "routing_policy": self.routing_policy,
            "fallback_available": self.fallback_available,
            "cloud_enabled": self.cloud_enabled,
            "detail": self.detail,
            "checked_at": self.checked_at,
        }


class LLMHealthChecker:
    """Performs Ollama health checks and model discovery.

    The checker depends on the provider registry (to obtain the Ollama client)
    and the model registry (to sync discovered models). It never depends on a
    specific provider SDK — it uses the :class:`LLMClient` interface.
    """

    def __init__(
        self,
        *,
        provider_registry: ProviderRegistry,
        model_registry: ModelRegistry,
        settings: LLMSettings,
        event_bus: EventBus | None = None,
    ) -> None:
        self._providers = provider_registry
        self._models = model_registry
        self._settings = settings
        self._event_bus = event_bus
        self._snapshot: LLMHealthSnapshot = LLMHealthSnapshot(
            provider=OLLAMA_PROVIDER_ID,
            status=HealthStatus.UNAVAILABLE.value,
            model=settings.ollama_default_model or settings.default_model,
            routing_policy=settings.routing_policy,
            fallback_available=True,
            cloud_enabled=settings.openai_compatible_configured,
            detail="Health check not yet performed.",
        )

    @property
    def snapshot(self) -> LLMHealthSnapshot:
        """The last health check result."""
        return self._snapshot

    async def check(self) -> LLMHealthSnapshot:
        """Run a health check + discovery and update the stored snapshot.

        Never raises. On any unexpected error, records
        ``INVALID_RESPONSE`` and returns.
        """
        model = self._settings.ollama_default_model or self._settings.default_model
        client = self._client()
        if client is None:
            self._set_snapshot(
                LLMHealthSnapshot(
                    provider=OLLAMA_PROVIDER_ID,
                    status=HealthStatus.UNAVAILABLE.value,
                    model=model,
                    routing_policy=self._settings.routing_policy,
                    fallback_available=True,
                    cloud_enabled=self._settings.openai_compatible_configured,
                    detail="Ollama provider not registered.",
                )
            )
            await self._publish_unavailable(model, "provider not registered")
            return self._snapshot

        try:
            health = await client.check_health(model)
        except Exception as exc:  # noqa: BLE001 - health checks must never raise
            logger.bind(
                event="llm.health.error", provider=OLLAMA_PROVIDER_ID, error=type(exc).__name__
            ).warning("Health check raised: {}", str(exc))
            self._set_snapshot(
                LLMHealthSnapshot(
                    provider=OLLAMA_PROVIDER_ID,
                    status=HealthStatus.INVALID_RESPONSE.value,
                    model=model,
                    routing_policy=self._settings.routing_policy,
                    fallback_available=True,
                    cloud_enabled=self._settings.openai_compatible_configured,
                    detail=f"{type(exc).__name__}: {str(exc)[:200]}",
                )
            )
            await self._publish_unavailable(model, "health check raised")
            return self._snapshot
        installed = await self._discover_installed_models(client)
        capabilities = await self._discover_capabilities(client, model, health)

        # Sync the discovered model into the registry.
        if health.available and model in installed:
            await self._sync_model(model, capabilities)

        snap = LLMHealthSnapshot(
            provider=OLLAMA_PROVIDER_ID,
            status=health.status,
            model=model,
            model_available=health.available,
            installed_models=installed,
            capabilities=capabilities,
            routing_policy=self._settings.routing_policy,
            fallback_available=True,
            cloud_enabled=self._settings.openai_compatible_configured,
            detail=health.detail,
        )
        self._set_snapshot(snap)

        if health.available:
            logger.bind(
                event="llm.health.ok",
                provider=OLLAMA_PROVIDER_ID,
                model=model,
                installed=len(installed),
                capabilities=capabilities,
            ).info("Ollama healthy: model '{}' available.", model)
        else:
            await self._publish_unavailable(model, health.detail or health.status)

        return self._snapshot

    def _client(self) -> LLMClient | None:
        if not self._providers.exists(OLLAMA_PROVIDER_ID):
            return None
        return self._providers.get(OLLAMA_PROVIDER_ID)

    async def _discover_installed_models(self, client: LLMClient) -> list[str]:
        """Return installed model names (best-effort, never raises)."""
        if not hasattr(client, "discover_models"):
            return []
        try:
            return list(await client.discover_models())
        except Exception as exc:  # noqa: BLE001 - discovery must never raise
            logger.bind(event="llm.discovery.error", error=type(exc).__name__).warning(
                "Model discovery failed: {}", str(exc)
            )
            return []

    async def _discover_capabilities(
        self, client: LLMClient, model: str, health: ProviderHealth
    ) -> list[str]:
        """Discover the capabilities of ``model`` via ``/api/show``.

        Falls back to the registry's existing definition when the API call
        fails or returns no info. Never raises.
        """
        if not hasattr(client, "get_model_info"):
            return [c.value for c in self._registry_capabilities(model)]
        try:
            info = await client.get_model_info(model)
        except Exception as exc:  # noqa: BLE001 - capability detection must never raise
            logger.bind(
                event="llm.capabilities.error", model=model, error=type(exc).__name__
            ).debug("Capability detection failed: {}", str(exc))
            return [c.value for c in self._registry_capabilities(model)]
        if info is None:
            # Provider could not report capabilities — use registry definition.
            return [c.value for c in self._registry_capabilities(model)]
        return sorted(c.value for c in info.capabilities)

    def _registry_capabilities(self, model: str) -> set[ModelCapability]:
        try:
            return self._models.get(OLLAMA_PROVIDER_ID, model).capabilities
        except Exception:  # noqa: BLE001
            return set()

    async def _sync_model(self, model: str, capabilities: list[str]) -> None:
        """Update the registry entry for ``model`` with discovered capabilities.

        Replaces the static bootstrap definition with one reflecting actual
        Ollama metadata (e.g. ``supports_tools`` when Ollama reports the
        ``tools`` capability). Preserves configured metadata (local, enabled).
        """
        cap_set = {ModelCapability(c) for c in capabilities if c}
        supports_tools = ModelCapability.TOOL_CALLING.value in capabilities
        supports_structured = (
            ModelCapability.STRUCTURED_OUTPUT.value in capabilities
            or supports_tools  # models with tools generally support JSON format
        )
        updated = ModelDefinition(
            model_id=model,
            provider=OLLAMA_PROVIDER_ID,
            display_name=model,
            capabilities=cap_set,
            context_window=32768 if "coder" in model else 8192,
            supports_structured_output=supports_structured,
            supports_tools=supports_tools,
            supports_reasoning=ModelCapability.REASONING.value in capabilities,
            coding_score=0.8 if "coder" in model else 0.5,
            local=True,
            enabled=True,
        )
        key = (OLLAMA_PROVIDER_ID, model)
        existing = self._models._models.get(key)  # noqa: SLF001 - registry mutation
        if existing is not None:
            self._models._models[key] = updated  # noqa: SLF001
        else:
            with contextlib.suppress(Exception):
                self._models.register(updated)
        logger.bind(
            event="llm.model.synced",
            model=model,
            supports_tools=supports_tools,
            capabilities=capabilities,
        ).info("Synced model '{}' from Ollama discovery.", model)

    def _set_snapshot(self, snapshot: LLMHealthSnapshot) -> None:
        from app.core.identifiers import utc_now

        snapshot.checked_at = utc_now().isoformat()
        self._snapshot = snapshot

    async def _publish_unavailable(self, model: str, detail: str) -> None:
        logger.bind(
            event="llm.health.degraded",
            provider=OLLAMA_PROVIDER_ID,
            model=model,
            detail=detail,
        ).warning("Ollama unavailable: {}", detail)
        if self._event_bus is None:
            return
        await self._event_bus.publish(
            Event.create(
                EventType.MODEL_UNAVAILABLE,
                payload={
                    "provider": OLLAMA_PROVIDER_ID,
                    "model": model,
                    "detail": detail,
                },
            )
        )
