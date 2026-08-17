"""Agent router: select the best agent definition for a task (Phase 7).

The router is *capability-oriented*: it scores candidate definitions against a
normalized task and selects the best active match. It may use the LLM for
semantic task→agent classification (structured output), but routing stays
deterministic enough to test without a real LLM — a deterministic
capability-based fallback is always available and is used whenever the LLM is
disabled or unavailable.

Selection inputs (capability-oriented, never a pile of if/else rules):

* task type / requested capabilities (matched against definition capabilities
  and task_types)
* available tools (definition's required/preferred tools must be registered)
* privacy requirements (definition privacy must satisfy task privacy)
* risk level (definition risk must be within the task's allowed risk)
* lifecycle state (only ACTIVE definitions are selectable by default)
* model requirements (best-effort; recorded, not blocking)
* historical performance (success rate / recent usage from execution records)
* availability (required tools registered)

The router never executes anything — it only selects. Execution is the
dispatcher + loop engine's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.agents.definitions import AgentDefinition
from app.agents.interface import AgentCapability
from app.core.exceptions import AgentRoutingError
from app.core.logging import get_logger
from app.events import Event, EventBus, EventType
from app.llm.models import PrivacyLevel
from app.tools.interface import RiskLevel

if TYPE_CHECKING:
    from app.agents.catalog import AgentDefinitionRegistry
    from app.agents.store import AgentExecutionStore
    from app.llm.router import ModelRouter
    from app.tools.registry import ToolRegistry

logger = get_logger("agents.router")

__all__ = [
    "RoutingTask",
    "RoutingDecision",
    "RoutingReason",
    "AgentRouter",
]


# ---------------------------------------------------------------------------
# Inputs / outputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoutingTask:
    """A normalized task the router selects an agent for."""

    goal: str
    task_type: str | None = None
    requested_capabilities: set[AgentCapability] = field(default_factory=set)
    required_tools: list[str] = field(default_factory=list)
    privacy: PrivacyLevel = PrivacyLevel.INTERNAL
    max_risk: RiskLevel = RiskLevel.MEDIUM
    agent_id: str | None = None  # explicit override
    metadata: dict[str, Any] = field(default_factory=dict)


class RoutingReason:
    """Why a routing decision was made (for observability)."""

    EXPLICIT = "explicit_agent_id"
    LLM_CLASSIFICATION = "llm_classification"
    CAPABILITY_MATCH = "capability_match"
    NO_MATCH = "no_match"
    LLM_UNAVAILABLE = "llm_unavailable"
    LLM_DISABLED = "llm_disabled"
    LLM_INVALID = "llm_invalid"


@dataclass(frozen=True)
class RoutingDecision:
    """Structured result of agent routing.

    ``agent_id`` is ``None`` when no suitable active agent exists — the
    orchestrator then asks the :class:`DynamicAgentBuilder` to create one.
    """

    agent_id: str | None
    reason: str
    score: float = 0.0
    candidates_considered: int = 0
    used_llm: bool = False
    fallback: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

_RISK_ORDER = {
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 1,
    RiskLevel.HIGH: 2,
    RiskLevel.CRITICAL: 3,
}


class AgentRouter:
    """Capability-oriented agent router with LLM classification + fallback."""

    def __init__(
        self,
        catalog: AgentDefinitionRegistry,
        tool_registry: ToolRegistry,
        *,
        event_bus: EventBus | None = None,
        model_router: ModelRouter | None = None,
        execution_store: AgentExecutionStore | None = None,
        llm_enabled: bool = False,
        min_match_score: float = 0.25,
    ) -> None:
        self._catalog = catalog
        self._tools = tool_registry
        self._event_bus = event_bus
        self._model_router = model_router
        self._execution_store = execution_store
        self._llm_enabled = llm_enabled
        # Below this deterministic score, no agent is considered a real match;
        # the orchestrator then asks the dynamic builder to create one.
        self._min_match_score = min_match_score

    async def route(self, task: RoutingTask) -> RoutingDecision:
        """Select an agent for ``task``.

        Order of precedence:
        1. Explicit ``task.agent_id`` (validated as dispatchable).
        2. LLM semantic classification (when enabled & available).
        3. Deterministic capability-based scoring.
        4. ``None`` (no match) → orchestrator triggers dynamic builder.
        """
        if task.agent_id:
            decision = await self._explicit(task)
            await self._publish(task, decision)
            return decision

        candidates = await self._eligible_candidates(task)
        if not candidates:
            decision = RoutingDecision(
                agent_id=None,
                reason=RoutingReason.NO_MATCH,
                candidates_considered=0,
            )
            await self._publish(task, decision)
            return decision

        # LLM classification (best-effort; fallback on any failure).
        if self._llm_enabled and self._model_router is not None:
            decision = await self._route_with_llm(task, candidates)
            if decision is not None:
                await self._publish(task, decision)
                return decision

        # Deterministic capability-based scoring.
        decision = self._score_candidates(task, candidates)
        if decision.score < self._min_match_score:
            decision = RoutingDecision(
                agent_id=None,
                reason=RoutingReason.NO_MATCH,
                candidates_considered=len(candidates),
                metadata={"best_score": decision.score, "best_agent": decision.agent_id},
            )
        await self._publish(task, decision)
        return decision

    # --- eligibility -----------------------------------------------------

    async def _eligible_candidates(self, task: RoutingTask) -> list[AgentDefinition]:
        """Active definitions satisfying privacy + risk + tool availability."""
        active = await self._catalog.list_active()
        eligible: list[AgentDefinition] = []
        for d in active:
            if not _privacy_ok(d, task):
                continue
            if not _risk_ok(d, task):
                continue
            if not self._tools_available(d):
                continue
            eligible.append(d)
        return eligible

    def _tools_available(self, d: AgentDefinition) -> bool:
        """All required tools must be registered (availability signal)."""
        return all(self._tools.exists(name) for name in d.required_tools)

    # --- explicit --------------------------------------------------------

    async def _explicit(self, task: RoutingTask) -> RoutingDecision:
        d = await self._catalog.find(task.agent_id or "")
        if d is None:
            raise AgentRoutingError(f"Explicit agent '{task.agent_id}' not found in catalog")
        if not d.is_dispatchable:
            raise AgentRoutingError(
                f"Agent '{d.agent_id}' is not dispatchable (lifecycle={d.lifecycle.value})"
            )
        return RoutingDecision(
            agent_id=d.agent_id,
            reason=RoutingReason.EXPLICIT,
            score=1.0,
            candidates_considered=1,
            metadata={"lifecycle": d.lifecycle.value, "version": d.version},
        )

    # --- LLM classification ---------------------------------------------

    async def _route_with_llm(
        self, task: RoutingTask, candidates: list[AgentDefinition]
    ) -> RoutingDecision | None:
        """Use the LLM to classify the task to one of the candidates.

        Returns None when the LLM is unavailable or returns an invalid/unknown
        agent, so the caller falls back to deterministic scoring.
        """
        from app.llm.models import LLMMessage, LLMRequest, MessageRole

        client = await self._select_client()
        if client is None:
            return None
        roster = [
            {
                "agent_id": d.agent_id,
                "name": d.name,
                "purpose": d.purpose,
                "capabilities": sorted(c.value for c in d.capabilities),
                "task_types": d.task_types,
            }
            for d in candidates
        ]
        messages = [
            LLMMessage(
                role=MessageRole.SYSTEM,
                content=(
                    "You are an agent router. Given a user goal and a roster of "
                    "agents, select the single best agent_id. Respond ONLY with a "
                    'JSON object: {"agent_id": "...", "reason": "..."}. '
                    "The agent_id MUST be one from the roster. Do not invent agents."
                ),
            ),
            LLMMessage(
                role=MessageRole.USER,
                content=(
                    f"Goal: {task.goal}\n"
                    f"Task type: {task.task_type or 'unspecified'}\n"
                    f"Requested capabilities: {sorted(c.value for c in task.requested_capabilities)}\n"
                    f"Agent roster: {roster}\n"
                    f"Select the best agent_id."
                ),
            ),
        ]
        try:
            request = LLMRequest(
                model=self._select_model_id(),
                messages=messages,
                temperature=0.0,
                privacy=task.privacy,
                metadata={"stage": "agent_routing"},
            )
            response = await client.generate(request)
        except Exception as exc:  # noqa: BLE001 - LLM unavailable → fallback
            logger.bind(event="agent.routing.llm.error", error=type(exc).__name__).debug(
                "LLM routing failed (falling back): {}", str(exc)
            )
            return None
        return self._parse_llm_decision(response, candidates)

    async def _select_client(self):
        """Resolve an LLM client via the model router (None if unavailable)."""
        if self._model_router is None:
            return None
        try:
            from app.llm.models import ModelCapability, PrivacyLevel
            from app.llm.router import RoutingRequest
        except ImportError:  # pragma: no cover
            return None
        request = RoutingRequest(
            capabilities={ModelCapability.CHAT},
            privacy=PrivacyLevel.INTERNAL,
            prefer_local=True,
            metadata={"stage": "agent_routing"},
        )
        try:
            selection = await self._model_router.select(request)
        except Exception:  # noqa: BLE001 - LLM unavailable → fallback
            return None
        providers = self._model_router._providers  # noqa: SLF001
        if not providers.exists(selection.provider):
            return None
        return providers.get(selection.provider)

    def _select_model_id(self) -> str:
        """Placeholder model id; providers resolve their default when empty."""
        return ""

    def _parse_llm_decision(
        self, response: Any, candidates: list[AgentDefinition]
    ) -> RoutingDecision | None:
        import json

        raw = response.content or ""
        try:
            # Tolerate fenced JSON.
            text = raw.strip()
            if text.startswith("```"):
                text = text.strip("`")
                if text.lower().startswith("json"):
                    text = text[4:]
                text = text.strip("`").strip()
            parsed = json.loads(text)
            agent_id = parsed.get("agent_id")
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(agent_id, str):
            return None
        match = next((d for d in candidates if d.agent_id == agent_id), None)
        if match is None:
            return None
        return RoutingDecision(
            agent_id=match.agent_id,
            reason=RoutingReason.LLM_CLASSIFICATION,
            score=1.0,
            candidates_considered=len(candidates),
            used_llm=True,
            metadata={"llm_reason": parsed.get("reason", "")},
        )

    # --- deterministic scoring ------------------------------------------

    def _score_candidates(
        self, task: RoutingTask, candidates: list[AgentDefinition]
    ) -> RoutingDecision:
        scored = [(d, self._score(d, task)) for d in candidates]
        scored.sort(key=lambda item: item[1], reverse=True)
        best, best_score = scored[0]
        return RoutingDecision(
            agent_id=best.agent_id,
            reason=RoutingReason.CAPABILITY_MATCH,
            score=best_score,
            candidates_considered=len(candidates),
            metadata={
                "version": best.version,
                "kind": best.kind.value,
            },
        )

    def _score(self, d: AgentDefinition, task: RoutingTask) -> float:
        """Deterministic capability-oriented score in [0, 1]."""
        score = 0.0
        # Capability overlap (most important signal).
        if task.requested_capabilities:
            req = set(task.requested_capabilities)
            overlap = len(req & d.capabilities) / len(req)
        else:
            overlap = 0.0
        score += 0.4 * overlap
        # Task type exact match.
        if task.task_type and task.task_type in d.task_types:
            score += 0.3
        elif task.task_type:
            # partial: task_type word overlap with purpose/name
            tl = task.task_type.lower()
            if tl in d.purpose.lower() or tl in d.name.lower():
                score += 0.1
        # Goal keyword overlap with capabilities/task_types/purpose.
        goal_l = task.goal.lower()
        for cap in d.capabilities:
            if cap.value in goal_l:
                score += 0.05
        for tt in d.task_types:
            if tt.lower() in goal_l:
                score += 0.05
        if d.purpose and any(w in goal_l for w in d.purpose.lower().split() if len(w) > 3):
            score += 0.05
        # Historical performance (small weight; never the sole driver).
        perf = self._performance_bonus(d)
        score += 0.1 * perf
        # Prefer preferred tools availability.
        if d.preferred_tools and all(self._tools.exists(t) for t in d.preferred_tools):
            score += 0.05
        return min(score, 1.0)

    def _performance_bonus(self, d: AgentDefinition) -> float:
        """Cheap success-rate proxy from usage metadata (no DB call)."""
        u = d.usage
        if u.dispatch_count == 0:
            return 0.5  # neutral for unproven agents
        return u.success_count / u.dispatch_count

    # --- events ---------------------------------------------------------

    async def _publish(self, task: RoutingTask, decision: RoutingDecision) -> None:
        if self._event_bus is None:
            return
        await self._event_bus.publish(
            Event.create(
                EventType.AGENT_SELECTED,
                agent_id=decision.agent_id,
                payload={
                    "reason": decision.reason,
                    "score": decision.score,
                    "candidates": decision.candidates_considered,
                    "used_llm": decision.used_llm,
                    "fallback": decision.fallback,
                },
                metadata={"reason": decision.reason},
            )
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PRIVACY_OK = {
    # definition privacy -> whether it may serve a task at this privacy level.
    # A PRIVATE/SENSITIVE task requires a definition that is itself non-public
    # (i.e., won't route to cloud). Definitions default to INTERNAL.
    PrivacyLevel.PUBLIC: {
        PrivacyLevel.PUBLIC,
        PrivacyLevel.INTERNAL,
        PrivacyLevel.PRIVATE,
        PrivacyLevel.SENSITIVE,
    },
    PrivacyLevel.INTERNAL: {PrivacyLevel.INTERNAL, PrivacyLevel.PRIVATE, PrivacyLevel.SENSITIVE},
    PrivacyLevel.PRIVATE: {PrivacyLevel.PRIVATE, PrivacyLevel.SENSITIVE},
    PrivacyLevel.SENSITIVE: {PrivacyLevel.SENSITIVE},
}


def _privacy_ok(d: AgentDefinition, task: RoutingTask) -> bool:
    allowed = _PRIVACY_OK.get(task.privacy, {PrivacyLevel.SENSITIVE})
    return d.privacy in allowed


def _risk_ok(d: AgentDefinition, task: RoutingTask) -> bool:
    return _RISK_ORDER[d.risk] <= _RISK_ORDER[task.max_risk]
