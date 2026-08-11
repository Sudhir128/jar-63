"""Phase 5 tests: LLM health checker and model discovery sync.

All tests are fully offline — they inject stub LLM clients that simulate
Ollama's health/model-info responses. No real Ollama server is required.
"""

from __future__ import annotations

from typing import Any

from app.config import LLMSettings
from app.events import EventType, InMemoryEventBus
from app.llm.client import HealthStatus, LLMClient, ProviderHealth
from app.llm.health import LLMHealthChecker
from app.llm.models import (
    LLMRequest,
    LLMResponse,
    ModelCapability,
    ModelDefinition,
)
from app.llm.providers.ollama import OLLAMA_PROVIDER_ID
from app.llm.registry import ModelRegistry, ProviderRegistry


class StubOllamaClient(LLMClient):
    """In-memory Ollama-like client for offline health-check tests.

    Simulates ``check_health``, ``discover_models``, and ``get_model_info``
    without any network access.
    """

    provider_name = OLLAMA_PROVIDER_ID

    def __init__(
        self,
        *,
        installed: list[str] | None = None,
        model_infos: dict[str, ModelDefinition] | None = None,
        health_status: str = HealthStatus.AVAILABLE.value,
        health_detail: str | None = None,
        raise_on_health: bool = False,
    ) -> None:
        self._installed = installed or []
        self._model_infos = model_infos or {}
        self._health_status = health_status
        self._health_detail = health_detail
        self._raise_on_health = raise_on_health
        self.closed = False

    async def generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(content="", model=request.model, provider=self.provider_name)

    async def generate_structured(
        self, request: LLMRequest, spec: Any | None = None
    ) -> LLMResponse:
        return LLMResponse(content="{}", model=request.model, provider=self.provider_name)

    async def check_health(self, model: str | None = None) -> ProviderHealth:
        if self._raise_on_health:
            raise RuntimeError("simulated health-check crash")
        if model is None:
            return ProviderHealth(
                status=self._health_status,
                provider=self.provider_name,
                detail=self._health_detail,
            )
        if self._health_status != HealthStatus.AVAILABLE.value:
            return ProviderHealth(
                status=self._health_status,
                provider=self.provider_name,
                model=model,
                detail=self._health_detail,
            )
        if model in self._installed:
            return ProviderHealth(
                status=HealthStatus.AVAILABLE, provider=self.provider_name, model=model
            )
        return ProviderHealth(
            status=HealthStatus.MODEL_NOT_FOUND,
            provider=self.provider_name,
            model=model,
            detail=f"Model '{model}' not installed.",
        )

    async def get_model_info(self, model: str) -> ModelDefinition | None:
        return self._model_infos.get(model)

    async def discover_models(self) -> list[str]:
        return list(self._installed)

    async def close(self) -> None:
        self.closed = True


def _qwen_info(model: str = "qwen2.5-coder:7b") -> ModelDefinition:
    return ModelDefinition(
        model_id=model,
        provider=OLLAMA_PROVIDER_ID,
        display_name=model,
        capabilities={
            ModelCapability.CHAT,
            ModelCapability.CODING,
            ModelCapability.TOOL_CALLING,
        },
        context_window=32768,
        supports_structured_output=True,
        supports_tools=True,
        local=True,
        enabled=True,
    )


def _settings(model: str = "qwen2.5-coder:7b") -> LLMSettings:
    return LLMSettings(
        _env_file=None,
        enabled=True,
        default_provider="ollama",
        default_model=model,
        ollama_default_model=model,
        routing_policy="local_first",
    )


def _build_checker(
    client: StubOllamaClient,
    *,
    settings: LLMSettings | None = None,
    event_bus: InMemoryEventBus | None = None,
    pre_register_model: ModelDefinition | None = None,
) -> tuple[LLMHealthChecker, ModelRegistry, ProviderRegistry]:
    settings = settings or _settings()
    models = ModelRegistry()
    providers = ProviderRegistry()
    if pre_register_model is not None:
        models.register(pre_register_model)
    providers.register(OLLAMA_PROVIDER_ID, client)
    bus = event_bus or InMemoryEventBus()
    checker = LLMHealthChecker(
        provider_registry=providers,
        model_registry=models,
        settings=settings,
        event_bus=bus,
    )
    return checker, models, providers


# ---------------------------------------------------------------------------
# Health check outcomes
# ---------------------------------------------------------------------------


async def test_health_available_default_model_installed() -> None:
    client = StubOllamaClient(
        installed=["qwen2.5-coder:7b"], model_infos={"qwen2.5-coder:7b": _qwen_info()}
    )
    checker, models, _ = _build_checker(client)
    snap = await checker.check()
    assert snap.available
    assert snap.status == HealthStatus.AVAILABLE.value
    assert snap.model == "qwen2.5-coder:7b"
    assert snap.model_available is True
    assert "qwen2.5-coder:7b" in snap.installed_models
    assert "tool_calling" in snap.capabilities


async def test_health_model_not_installed() -> None:
    client = StubOllamaClient(installed=[], model_infos={})
    checker, _, _ = _build_checker(client)
    snap = await checker.check()
    assert not snap.available
    assert snap.degraded
    assert snap.status == HealthStatus.MODEL_NOT_FOUND.value
    assert snap.model_available is False
    assert "not installed" in (snap.detail or "").lower()


async def test_health_provider_unavailable() -> None:
    client = StubOllamaClient(health_status=HealthStatus.UNAVAILABLE.value)
    checker, _, _ = _build_checker(client)
    snap = await checker.check()
    assert not snap.available
    assert snap.status == HealthStatus.UNAVAILABLE.value


async def test_health_provider_timeout() -> None:
    client = StubOllamaClient(health_status=HealthStatus.TIMEOUT.value)
    checker, _, _ = _build_checker(client)
    snap = await checker.check()
    assert not snap.available
    assert snap.status == HealthStatus.TIMEOUT.value


async def test_health_invalid_response() -> None:
    client = StubOllamaClient(health_status=HealthStatus.INVALID_RESPONSE.value)
    checker, _, _ = _build_checker(client)
    snap = await checker.check()
    assert not snap.available
    assert snap.status == HealthStatus.INVALID_RESPONSE.value


async def test_health_never_raises_on_client_crash() -> None:
    client = StubOllamaClient(raise_on_health=True)
    checker, _, _ = _build_checker(client)
    # Must not raise.
    snap = await checker.check()
    assert not snap.available


async def test_health_provider_not_registered() -> None:
    settings = _settings()
    models = ModelRegistry()
    providers = ProviderRegistry()  # no Ollama registered
    checker = LLMHealthChecker(
        provider_registry=providers,
        model_registry=models,
        settings=settings,
        event_bus=InMemoryEventBus(),
    )
    snap = await checker.check()
    assert not snap.available
    assert snap.status == HealthStatus.UNAVAILABLE.value
    assert "not registered" in (snap.detail or "")


# ---------------------------------------------------------------------------
# Discovery + registry sync
# ---------------------------------------------------------------------------


async def test_discovery_syncs_capabilities_into_registry() -> None:
    # Bootstrap registers a static model WITHOUT tool_calling.
    static = ModelDefinition(
        model_id="qwen2.5-coder:7b",
        provider=OLLAMA_PROVIDER_ID,
        display_name="qwen2.5-coder:7b",
        capabilities={ModelCapability.CHAT, ModelCapability.CODING},
        supports_tools=False,
        local=True,
        enabled=True,
    )
    client = StubOllamaClient(
        installed=["qwen2.5-coder:7b"], model_infos={"qwen2.5-coder:7b": _qwen_info()}
    )
    checker, models, _ = _build_checker(client, pre_register_model=static)
    await checker.check()
    synced = models.get(OLLAMA_PROVIDER_ID, "qwen2.5-coder:7b")
    # Discovery should have added tool_calling.
    assert ModelCapability.TOOL_CALLING in synced.capabilities
    assert synced.supports_tools is True


async def test_discovery_does_not_sync_when_model_missing() -> None:
    static = ModelDefinition(
        model_id="qwen2.5-coder:7b",
        provider=OLLAMA_PROVIDER_ID,
        capabilities={ModelCapability.CHAT},
        supports_tools=False,
        local=True,
        enabled=True,
    )
    client = StubOllamaClient(installed=[])
    checker, models, _ = _build_checker(client, pre_register_model=static)
    await checker.check()
    synced = models.get(OLLAMA_PROVIDER_ID, "qwen2.5-coder:7b")
    # Unchanged — no sync.
    assert ModelCapability.TOOL_CALLING not in synced.capabilities
    assert synced.supports_tools is False


async def test_discovery_lists_all_installed_models() -> None:
    client = StubOllamaClient(
        installed=["qwen2.5-coder:7b", "llama3.2:3b", "nomic-embed-text"],
        model_infos={"qwen2.5-coder:7b": _qwen_info()},
    )
    checker, _, _ = _build_checker(client)
    snap = await checker.check()
    assert set(snap.installed_models) == {"qwen2.5-coder:7b", "llama3.2:3b", "nomic-embed-text"}


async def test_discovery_capability_fallback_to_registry() -> None:
    """When get_model_info fails, capabilities come from the registry."""
    static = ModelDefinition(
        model_id="qwen2.5-coder:7b",
        provider=OLLAMA_PROVIDER_ID,
        capabilities={ModelCapability.CHAT, ModelCapability.CODING},
        local=True,
        enabled=True,
    )
    client = StubOllamaClient(
        installed=["qwen2.5-coder:7b"],
        model_infos={},  # no info -> get_model_info returns None
    )
    checker, models, _ = _build_checker(client, pre_register_model=static)
    snap = await checker.check()
    # Capabilities fall back to registry (CHAT, CODING).
    assert "chat" in snap.capabilities
    assert "coding" in snap.capabilities


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


async def test_unavailable_publishes_model_unavailable_event() -> None:
    bus = InMemoryEventBus()
    events: list = []

    async def handler(ev) -> None:
        events.append(ev)

    bus.subscribe(EventType.MODEL_UNAVAILABLE, handler)
    client = StubOllamaClient(installed=[])
    checker, _, _ = _build_checker(client, event_bus=bus)
    await checker.check()
    assert any(ev.event_type is EventType.MODEL_UNAVAILABLE for ev in events)


async def test_available_does_not_publish_unavailable_event() -> None:
    bus = InMemoryEventBus()
    events: list = []

    async def handler(ev) -> None:
        events.append(ev)

    bus.subscribe(EventType.MODEL_UNAVAILABLE, handler)
    client = StubOllamaClient(
        installed=["qwen2.5-coder:7b"], model_infos={"qwen2.5-coder:7b": _qwen_info()}
    )
    checker, _, _ = _build_checker(client, event_bus=bus)
    await checker.check()
    assert events == []


# ---------------------------------------------------------------------------
# Snapshot serialization
# ---------------------------------------------------------------------------


def test_snapshot_to_api_dict_has_no_secrets() -> None:
    client = StubOllamaClient(
        installed=["qwen2.5-coder:7b"], model_infos={"qwen2.5-coder:7b": _qwen_info()}
    )
    checker, _, _ = _build_checker(client)
    import json

    d = checker.snapshot.to_api_dict()
    blob = json.dumps(d)
    # No secret-like keys.
    assert "api_key" not in blob
    assert "authorization" not in blob
    assert "token" not in blob.lower()
    # Required fields present.
    assert {"status", "provider", "model", "model_available", "capabilities"} <= set(d)


def test_snapshot_degraded_property() -> None:
    client = StubOllamaClient(health_status=HealthStatus.UNAVAILABLE.value)
    checker, _, _ = _build_checker(client)
    # Before check: default snapshot is UNAVAILABLE -> degraded.
    assert checker.snapshot.degraded
    assert not checker.snapshot.available
