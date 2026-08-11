"""Tests for ModelRegistry and ProviderRegistry."""

from __future__ import annotations

import pytest

from app.llm.client import HealthStatus, LLMClient, ProviderHealth
from app.llm.models import LLMRequest, LLMResponse, ModelCapability, ModelDefinition
from app.llm.registry import (
    ModelAlreadyRegisteredError,
    ModelNotFoundError_,
    ModelRegistry,
    ProviderAlreadyRegisteredError,
    ProviderNotFoundError,
    ProviderRegistry,
)
from tests.llm_helpers import StubLLMClient


def _model(model_id: str = "m1", provider: str = "ollama", **kw) -> ModelDefinition:
    return ModelDefinition(model_id=model_id, provider=provider, **kw)


# --- ModelRegistry ---


def test_model_registry_register_and_get() -> None:
    reg = ModelRegistry()
    m = _model(capabilities={ModelCapability.CHAT})
    reg.register(m)
    assert reg.exists("ollama", "m1")
    assert reg.get("ollama", "m1") is m
    assert len(reg) == 1


def test_model_registry_duplicate_raises() -> None:
    reg = ModelRegistry()
    reg.register(_model())
    with pytest.raises(ModelAlreadyRegisteredError):
        reg.register(_model())


def test_model_registry_unregister() -> None:
    reg = ModelRegistry()
    reg.register(_model())
    reg.unregister("ollama", "m1")
    assert not reg.exists("ollama", "m1")


def test_model_registry_unregister_missing_raises() -> None:
    reg = ModelRegistry()
    with pytest.raises(ModelNotFoundError_):
        reg.unregister("ollama", "missing")


def test_model_registry_get_missing_raises() -> None:
    reg = ModelRegistry()
    with pytest.raises(ModelNotFoundError_):
        reg.get("ollama", "missing")


def test_model_registry_list() -> None:
    reg = ModelRegistry()
    reg.register(_model("a", capabilities={ModelCapability.CHAT}))
    reg.register(_model("b", capabilities={ModelCapability.CODING}))
    assert {m.model_id for m in reg.list()} == {"a", "b"}


def test_model_registry_find_by_capability() -> None:
    reg = ModelRegistry()
    reg.register(_model("a", capabilities={ModelCapability.CHAT, ModelCapability.CODING}))
    reg.register(_model("b", capabilities={ModelCapability.CHAT}))
    coding = reg.find_by_capability(ModelCapability.CODING)
    assert len(coding) == 1
    assert coding[0].model_id == "a"


def test_model_registry_enable_disable() -> None:
    reg = ModelRegistry()
    reg.register(_model())
    assert reg.get("ollama", "m1").enabled is True
    reg.disable("ollama", "m1")
    assert reg.get("ollama", "m1").enabled is False
    assert reg.list_enabled() == []
    reg.enable("ollama", "m1")
    assert reg.get("ollama", "m1").enabled is True


def test_model_registry_same_id_different_providers() -> None:
    reg = ModelRegistry()
    reg.register(_model("shared", provider="ollama"))
    reg.register(_model("shared", provider="openai_compatible"))
    assert reg.exists("ollama", "shared")
    assert reg.exists("openai_compatible", "shared")
    assert len(reg) == 2


# --- ProviderRegistry ---


async def test_provider_registry_register_and_get() -> None:
    reg = ProviderRegistry()
    client = StubLLMClient("ollama")
    reg.register("ollama", client)
    assert reg.exists("ollama")
    assert reg.get("ollama") is client


def test_provider_registry_duplicate_raises() -> None:
    reg = ProviderRegistry()
    reg.register("ollama", StubLLMClient("ollama"))
    with pytest.raises(ProviderAlreadyRegisteredError):
        reg.register("ollama", StubLLMClient("ollama"))


def test_provider_registry_get_missing_raises() -> None:
    reg = ProviderRegistry()
    with pytest.raises(ProviderNotFoundError):
        reg.get("missing")


def test_provider_registry_list() -> None:
    reg = ProviderRegistry()
    reg.register("ollama", StubLLMClient("ollama"))
    reg.register("openai_compatible", StubLLMClient("openai_compatible"))
    assert set(reg.list()) == {"ollama", "openai_compatible"}


async def test_provider_registry_health_check_available() -> None:
    reg = ProviderRegistry()
    reg.register("ollama", StubLLMClient("ollama"))
    health = await reg.health_check("ollama", "qwen2.5-coder:7b")
    assert health.available


async def test_provider_registry_health_check_unregistered() -> None:
    reg = ProviderRegistry()
    health = await reg.health_check("missing")
    assert health.status == HealthStatus.UNAVAILABLE
    assert "not registered" in (health.detail or "")


async def test_provider_registry_health_check_handles_exceptions() -> None:
    class ExplodingClient(LLMClient):
        @property
        def provider_name(self) -> str:
            return "boom"

        async def generate(self, request: LLMRequest) -> LLMResponse:
            raise RuntimeError

        async def generate_structured(self, request, spec=None) -> LLMResponse:
            raise RuntimeError

        async def check_health(self, model: str | None = None) -> ProviderHealth:
            raise RuntimeError("boom")

        async def get_model_info(self, model: str):
            return None

        async def close(self) -> None:
            pass

    reg = ProviderRegistry()
    reg.register("boom", ExplodingClient())
    health = await reg.health_check("boom")
    assert health.status == HealthStatus.UNAVAILABLE
