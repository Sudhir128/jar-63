"""Tests for the deterministic model router (local-first, privacy, offline)."""

from __future__ import annotations

import pytest

from app.llm.errors import ModelUnavailableError
from app.llm.models import ModelCapability, PrivacyLevel
from app.llm.registry import ModelRegistry, ProviderRegistry
from app.llm.router import (
    AlwaysOfflineNetworkChecker,
    ModelRouter,
    NetworkChecker,
    RoutingRequest,
)
from tests.llm_helpers import (
    StubLLMClient,
    cloud_coding_model,
    local_coding_model,
    make_llm_settings,
)


class OnlineNetwork(NetworkChecker):
    async def is_available(self, provider_id: str) -> bool:
        return True


def _build_router(
    *,
    models=None,
    providers=None,
    network=None,
    settings=None,
):
    mr = ModelRegistry()
    for m in models or []:
        mr.register(m)
    pr = ProviderRegistry()
    for pid, c in providers or []:
        pr.register(pid, c)
    return ModelRouter(
        model_registry=mr,
        provider_registry=pr,
        settings=settings or make_llm_settings(),
        network_checker=network or AlwaysOfflineNetworkChecker(),
    )


# --- Demo 1: local selection ---


async def test_router_selects_local_for_coding() -> None:
    router = _build_router(
        models=[local_coding_model()],
        providers=[("ollama", StubLLMClient("ollama"))],
    )
    sel = await router.select(
        RoutingRequest(capabilities={ModelCapability.CODING}, privacy=PrivacyLevel.PUBLIC)
    )
    assert sel.local
    assert sel.provider == "ollama"
    assert sel.reason  # non-empty reason


# --- Demo 2: cloud fallback ---


async def test_router_cloud_fallback_when_no_local() -> None:
    router = _build_router(
        models=[cloud_coding_model()],
        providers=[("openai_compatible", StubLLMClient("openai_compatible"))],
        network=OnlineNetwork(),
    )
    sel = await router.select(
        RoutingRequest(capabilities={ModelCapability.CODING}, privacy=PrivacyLevel.PUBLIC)
    )
    assert not sel.local
    assert sel.provider == "openai_compatible"


# --- Demo 3: offline + local available ---


async def test_router_offline_uses_local() -> None:
    router = _build_router(
        models=[local_coding_model()],
        providers=[("ollama", StubLLMClient("ollama"))],
        network=AlwaysOfflineNetworkChecker(),
    )
    sel = await router.select(
        RoutingRequest(capabilities={ModelCapability.CODING}, privacy=PrivacyLevel.PUBLIC)
    )
    assert sel.local
    assert sel.provider == "ollama"


# --- Demo 4: offline + no local model ---


async def test_router_offline_no_model_raises() -> None:
    router = _build_router(
        models=[cloud_coding_model()],
        providers=[("openai_compatible", StubLLMClient("openai_compatible"))],
        network=AlwaysOfflineNetworkChecker(),
    )
    with pytest.raises(ModelUnavailableError):
        await router.select(
            RoutingRequest(capabilities={ModelCapability.CODING}, privacy=PrivacyLevel.PUBLIC)
        )


# --- Demo 5: private task prefers local ---


async def test_router_private_prefers_local() -> None:
    router = _build_router(
        models=[local_coding_model(), cloud_coding_model()],
        providers=[
            ("ollama", StubLLMClient("ollama")),
            ("openai_compatible", StubLLMClient("openai_compatible")),
        ],
        network=AlwaysOfflineNetworkChecker(),
    )
    sel = await router.select(
        RoutingRequest(capabilities={ModelCapability.CODING}, privacy=PrivacyLevel.PRIVATE)
    )
    assert sel.local
    assert sel.provider == "ollama"


async def test_router_private_disallows_cloud_when_no_local() -> None:
    router = _build_router(
        models=[cloud_coding_model()],
        providers=[("openai_compatible", StubLLMClient("openai_compatible"))],
        network=OnlineNetwork(),
    )
    with pytest.raises(ModelUnavailableError, match="disallowed"):
        await router.select(
            RoutingRequest(capabilities={ModelCapability.CODING}, privacy=PrivacyLevel.PRIVATE)
        )


async def test_router_private_allows_cloud_when_configured() -> None:
    router = _build_router(
        models=[cloud_coding_model()],
        providers=[("openai_compatible", StubLLMClient("openai_compatible"))],
        network=OnlineNetwork(),
        settings=make_llm_settings(allow_cloud_private=True),
    )
    sel = await router.select(
        RoutingRequest(capabilities={ModelCapability.CODING}, privacy=PrivacyLevel.PRIVATE)
    )
    assert not sel.local
    assert sel.provider == "openai_compatible"


async def test_router_local_only_policy() -> None:
    settings = make_llm_settings()
    settings = settings.model_copy(update={"routing_policy": "local_only"})
    router = _build_router(
        models=[cloud_coding_model()],
        providers=[("openai_compatible", StubLLMClient("openai_compatible"))],
        network=OnlineNetwork(),
        settings=settings,
    )
    with pytest.raises(ModelUnavailableError, match="LOCAL_ONLY"):
        await router.select(
            RoutingRequest(capabilities={ModelCapability.CODING}, privacy=PrivacyLevel.PUBLIC)
        )


async def test_router_capability_filtering() -> None:
    # A model without CODING should not be selected for a coding request.
    from app.llm.models import ModelDefinition

    chat_only = ModelDefinition(
        model_id="chat-only:7b",
        provider="ollama",
        capabilities={ModelCapability.CHAT},
        local=True,
        enabled=True,
    )
    router = _build_router(
        models=[chat_only],
        providers=[("ollama", StubLLMClient("ollama"))],
    )
    with pytest.raises(ModelUnavailableError):
        await router.select(
            RoutingRequest(capabilities={ModelCapability.CODING}, privacy=PrivacyLevel.PUBLIC)
        )


async def test_router_disabled_models_skipped() -> None:
    from app.llm.models import ModelDefinition

    m = ModelDefinition(
        model_id="disabled:7b",
        provider="ollama",
        capabilities={ModelCapability.CODING},
        local=True,
        enabled=False,
    )
    router = _build_router(
        models=[m],
        providers=[("ollama", StubLLMClient("ollama"))],
    )
    with pytest.raises(ModelUnavailableError):
        await router.select(
            RoutingRequest(capabilities={ModelCapability.CODING}, privacy=PrivacyLevel.PUBLIC)
        )


async def test_router_context_filtering() -> None:
    small = local_coding_model()
    small = small.model_copy(update={"context_window": 4096})
    router = _build_router(
        models=[small],
        providers=[("ollama", StubLLMClient("ollama"))],
    )
    with pytest.raises(ModelUnavailableError):
        await router.select(
            RoutingRequest(
                capabilities={ModelCapability.CODING},
                privacy=PrivacyLevel.PUBLIC,
                context_tokens=8192,
            )
        )


async def test_router_network_is_signal_not_rule() -> None:
    """Internet available does NOT force cloud when a local model exists."""
    router = _build_router(
        models=[local_coding_model()],
        providers=[("ollama", StubLLMClient("ollama"))],
        network=OnlineNetwork(),
    )
    sel = await router.select(
        RoutingRequest(capabilities={ModelCapability.CODING}, privacy=PrivacyLevel.PUBLIC)
    )
    assert sel.local  # local-first wins despite network being up


async def test_router_publishes_selection_event(captured_events) -> None:
    bus, events = captured_events
    from app.events import EventType

    router = _build_router(
        models=[local_coding_model()],
        providers=[("ollama", StubLLMClient("ollama"))],
    )
    router._event_bus = bus  # noqa: SLF001
    await router.select(
        RoutingRequest(capabilities={ModelCapability.CODING}, privacy=PrivacyLevel.PUBLIC)
    )
    assert any(e.event_type is EventType.MODEL_SELECTED for e in events)


async def test_router_publishes_unavailable_event(captured_events) -> None:
    bus, events = captured_events
    from app.events import EventType

    router = _build_router(models=[], providers=[])
    router._event_bus = bus  # noqa: SLF001
    with pytest.raises(ModelUnavailableError):
        await router.select(
            RoutingRequest(capabilities={ModelCapability.CODING}, privacy=PrivacyLevel.PUBLIC)
        )
    assert any(e.event_type is EventType.MODEL_UNAVAILABLE for e in events)
