"""Deterministic model router.

The router selects a model based primarily on **task requirements**, not on
network availability alone. "Internet available → cloud" is explicitly NOT the
strategy. The default policy is ``LOCAL_FIRST``.

Selection inputs (Phase 2, deterministic):

* required capabilities
* privacy level
* model capabilities / availability
* local vs cloud availability
* provider availability
* user policy (allow cloud for private?)

The router returns a structured :class:`ModelSelection` (never an arbitrary
string) or raises :class:`ModelRoutingError`/``ModelUnavailableError`` when no
model can satisfy the request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from app.config import LLMSettings
from app.core.logging import get_logger
from app.events import Event, EventBus, EventType
from app.llm.errors import LLMPolicyError, ModelUnavailableError
from app.llm.models import ModelCapability, ModelDefinition, PrivacyLevel
from app.llm.registry import ModelRegistry, ProviderRegistry

logger = get_logger("llm.router")

__all__ = [
    "RoutingPolicy",
    "ModelSelection",
    "RoutingRequest",
    "NetworkChecker",
    "AlwaysOfflineNetworkChecker",
    "ModelRouter",
]


class RoutingPolicy(StrEnum):
    """Deterministic routing policies.

    Phase 2 ships ``LOCAL_FIRST``. Others are placeholders for future phases.
    """

    LOCAL_FIRST = "local_first"
    LOCAL_ONLY = "local_only"
    CLOUD_FIRST = "cloud_first"


@dataclass(frozen=True)
class ModelSelection:
    """Structured result of model selection.

    Carries enough decision metadata for debugging and observability.
    """

    model_id: str
    provider: str
    local: bool
    reason: str
    capabilities_met: list[str] = field(default_factory=list)
    privacy_ok: bool = True
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def display(self) -> str:
        return f"{self.provider}/{self.model_id}"


@dataclass(frozen=True)
class RoutingRequest:
    """Inputs describing what a task needs from a model."""

    capabilities: set[ModelCapability] = field(default_factory=set)
    privacy: PrivacyLevel = PrivacyLevel.PUBLIC
    prefer_local: bool = True
    context_tokens: int | None = None
    metadata: dict[str, str] = field(default_factory=dict)


class NetworkChecker:
    """Interface for probing remote provider reachability.

    Network availability is a *signal*, not the primary routing rule. The
    default implementation reports offline (so local-first routing is
    exercised). Real checks can be injected later.
    """

    async def is_available(self, provider_id: str) -> bool:
        raise NotImplementedError


class AlwaysOfflineNetworkChecker(NetworkChecker):
    """Reports every provider as unreachable (deterministic for tests)."""

    async def is_available(self, provider_id: str) -> bool:
        return False


class ModelRouter:
    """Selects a model for a routing request using a deterministic policy.

    The router depends on the registries, an optional network checker, and
    settings — never on provider SDKs.
    """

    def __init__(
        self,
        *,
        model_registry: ModelRegistry,
        provider_registry: ProviderRegistry,
        settings: LLMSettings,
        network_checker: NetworkChecker | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self._models = model_registry
        self._providers = provider_registry
        self._settings = settings
        self._network = network_checker or AlwaysOfflineNetworkChecker()
        self._event_bus = event_bus
        try:
            self._policy = RoutingPolicy(self._settings.routing_policy)
        except ValueError:
            self._policy = RoutingPolicy.LOCAL_FIRST

    async def select(self, request: RoutingRequest) -> ModelSelection:
        """Select the best model for ``request``.

        Raises :class:`ModelUnavailableError` if no model can satisfy the
        request under the active policy.
        """
        if self._policy is RoutingPolicy.LOCAL_ONLY:
            return await self._select_local_only(request)
        # LOCAL_FIRST and CLOUD_FIRST both consider local then cloud, with
        # privacy gating. (CLOUD_FIRST would reorder; Phase 2 keeps local-first
        # semantics for both to remain deterministic and safe.)
        return await self._select_local_first(request)

    async def _select_local_only(self, request: RoutingRequest) -> ModelSelection:
        local = await self._best_local(request)
        if local is not None:
            return local
        await self._publish_unavailable(request, reason="no local model satisfies the request")
        raise ModelUnavailableError("No local model satisfies the request (LOCAL_ONLY policy).")

    async def _select_local_first(self, request: RoutingRequest) -> ModelSelection:
        # Privacy gate: private/sensitive requests cannot use cloud unless allowed.
        cloud_allowed = (
            not request.privacy.is_cloud_restricted or self._settings.allow_cloud_for_private
        )

        local = await self._best_local(request)
        if local is not None:
            await self._publish_selected(local, request, source="local")
            return local

        if cloud_allowed:
            cloud = await self._best_cloud(request)
            if cloud is not None:
                await self._publish_selected(cloud, request, source="cloud")
                return cloud

        reason = (
            "No suitable local model and cloud use is disallowed for "
            f"{request.privacy.value} requests."
            if not cloud_allowed
            else "No suitable local or cloud model available."
        )
        await self._publish_unavailable(request, reason=reason)
        raise ModelUnavailableError(reason)

    async def _best_local(self, request: RoutingRequest) -> ModelSelection | None:
        local_models = [
            m
            for m in self._models.find_enabled_by_capability(_any(request.capabilities))
            if m.local
        ]
        local_models = _filter_capabilities(local_models, request.capabilities)
        local_models = _filter_context(local_models, request.context_tokens)
        if not local_models:
            return None
        # Prefer models whose provider is reachable, but do not require a live
        # health check for every selection (network is a signal, not a rule).
        model = _pick_best(local_models)
        if not self._providers.exists(model.provider):
            return None
        return ModelSelection(
            model_id=model.model_id,
            provider=model.provider,
            local=True,
            reason=(
                f"Local {model.display} matches capabilities "
                f"{[c.value for c in request.capabilities]} (local-first policy)."
            ),
            capabilities_met=[c.value for c in (request.capabilities & model.capabilities)],
            privacy_ok=True,
            metadata={"policy": self._policy.value, "source": "local"},
        )

    async def _best_cloud(self, request: RoutingRequest) -> ModelSelection | None:
        cloud_models = [
            m
            for m in self._models.find_enabled_by_capability(_any(request.capabilities))
            if not m.local
        ]
        cloud_models = _filter_capabilities(cloud_models, request.capabilities)
        cloud_models = _filter_context(cloud_models, request.context_tokens)
        if not cloud_models:
            return None
        # Network is a signal: only consider cloud if the provider is reachable.
        for model in _rank_cloud(cloud_models):
            if not self._providers.exists(model.provider):
                continue
            if await self._network.is_available(model.provider):
                return ModelSelection(
                    model_id=model.model_id,
                    provider=model.provider,
                    local=False,
                    reason=(
                        f"Cloud {model.display} selected as fallback "
                        f"(no suitable local model; network available)."
                    ),
                    capabilities_met=[c.value for c in (request.capabilities & model.capabilities)],
                    privacy_ok=True,
                    metadata={"policy": self._policy.value, "source": "cloud"},
                )
        return None

    async def _publish_selected(
        self, selection: ModelSelection, request: RoutingRequest, *, source: str
    ) -> None:
        logger.bind(
            event="model.selected",
            provider=selection.provider,
            model=selection.model_id,
            local=selection.local,
            source=source,
            privacy=request.privacy.value,
        ).info("Selected model: {}", selection.display)
        if self._event_bus is None:
            return
        await self._event_bus.publish(
            Event.create(
                EventType.MODEL_SELECTED,
                payload={
                    "provider": selection.provider,
                    "model": selection.model_id,
                    "local": selection.local,
                    "source": source,
                    "reason": selection.reason,
                    "privacy": request.privacy.value,
                },
            )
        )

    async def _publish_unavailable(self, request: RoutingRequest, *, reason: str) -> None:
        logger.bind(
            event="model.unavailable",
            privacy=request.privacy.value,
            capabilities=[c.value for c in request.capabilities],
        ).warning("No model available: {}", reason)
        if self._event_bus is None:
            return
        await self._event_bus.publish(
            Event.create(
                EventType.MODEL_UNAVAILABLE,
                payload={
                    "reason": reason,
                    "privacy": request.privacy.value,
                    "capabilities": [c.value for c in request.capabilities],
                },
            )
        )


def _any(caps: set[ModelCapability]) -> ModelCapability:
    """Return a representative capability for the registry query.

    The registry's ``find_enabled_by_capability`` takes a single capability;
    we query with an arbitrary one then filter precisely afterwards. If the
    set is empty we use CHAT (the baseline capability every model has).
    """
    if not caps:
        return ModelCapability.CHAT
    return next(iter(caps))


def _filter_capabilities(
    models: list[ModelDefinition], required: set[ModelCapability]
) -> list[ModelDefinition]:
    if not required:
        return models
    return [m for m in models if required.issubset(m.capabilities)]


def _filter_context(
    models: list[ModelDefinition], context_tokens: int | None
) -> list[ModelDefinition]:
    if context_tokens is None:
        return models
    return [m for m in models if m.context_window is None or m.context_window >= context_tokens]


def _pick_best(models: list[ModelDefinition]) -> ModelDefinition:
    """Deterministic tie-break: prefer higher coding_score, then display name."""
    return sorted(
        models,
        key=lambda m: (-(m.coding_score or 0.0), m.display),
    )[0]


def _rank_cloud(models: list[ModelDefinition]) -> list[ModelDefinition]:
    return sorted(models, key=lambda m: (-(m.coding_score or 0.0), m.display))


def assert_policy_allows_cloud(settings: LLMSettings, privacy: PrivacyLevel) -> None:
    """Raise :class:`LLMPolicyError` if cloud use is disallowed for ``privacy``."""
    if privacy.is_cloud_restricted and not settings.allow_cloud_for_private:
        raise LLMPolicyError(f"Cloud providers are disallowed for {privacy.value} requests.")
