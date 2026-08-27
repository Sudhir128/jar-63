"""Dynamic agent builder (Phase 7).

When the router finds no suitable active agent, the orchestrator may ask the
:class:`DynamicAgentBuilder` to create a new agent definition — controlled,
validated, and bounded — instead of failing.

Flow::

    TASK
      ↓
    ROUTER (no suitable agent)
      ↓
    DynamicAgentBuilder
      ↓
    LLM → structured AgentSpec (validated)
      ↓
    validation (malformed/unsafe/impossible/contradictory rejected)
      ↓
    AgentDefinition
      ↓
    catalog.register
      ↓
    dispatch (by the orchestrator/dispatcher)

The LLM never writes Python or arbitrary executable code. It only produces a
*structured* :class:`AgentSpec` (purpose, capabilities, task types, tools,
model/privacy/risk requirements, instructions, constraints, verification
strategy). The spec is validated before registration.

Safety:
* Dynamic agents are always ``GENERIC`` kind — their instructions guide
  behavior, but they execute *only* through the :class:`ToolExecutor` and
  :class:`ToolPolicy`.
* CRITICAL-risk dynamic agents are rejected outright (never auto-active).
* HIGH-risk dynamic agents are created INACTIVE (require manual activation).
* Required tools must already exist in the :class:`ToolRegistry`; the builder
  never registers new/unrestricted tools.
* The number of dynamic agents is capped by ``max_dynamic_agents``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.agents.definitions import (
    AgentDefinition,
    AgentKind,
    AgentLifecycleState,
    AgentSpec,
)
from app.agents.interface import AgentCapability
from app.core.exceptions import AgentDefinitionValidationError
from app.core.identifiers import generate_id, utc_now
from app.core.logging import get_logger
from app.events import Event, EventBus, EventType
from app.llm.models import PrivacyLevel
from app.tools.interface import RiskLevel

if TYPE_CHECKING:
    from app.agents.catalog import AgentDefinitionRegistry
    from app.llm.router import ModelRouter
    from app.tools.registry import ToolRegistry

logger = get_logger("agents.dynamic_builder")

__all__ = ["DynamicAgentBuilder", "BuildResult", "BuildFallbackReason"]


class BuildFallbackReason:
    """Why the builder used the deterministic spec instead of the LLM."""

    LLM_DISABLED = "llm_disabled"
    NO_MODEL = "no_model"
    LLM_ERROR = "llm_error"
    INVALID_SPEC = "invalid_spec"


@dataclass(frozen=True)
class BuildResult:
    """The outcome of a dynamic build attempt."""

    definition: AgentDefinition
    created: bool  # True if newly registered; False if an existing one was reused
    used_llm: bool
    fallback_reason: str | None = None
    metadata: dict[str, Any] | None = None


class DynamicAgentBuilder:
    """Controlled builder of dynamic agent definitions."""

    def __init__(
        self,
        catalog: AgentDefinitionRegistry,
        tool_registry: ToolRegistry,
        *,
        event_bus: EventBus | None = None,
        model_router: ModelRouter | None = None,
        llm_enabled: bool = False,
        max_dynamic_agents: int = 50,
        auto_activate_dynamic: bool = True,
    ) -> None:
        self._catalog = catalog
        self._tools = tool_registry
        self._event_bus = event_bus
        self._router = model_router
        self._llm_enabled = llm_enabled
        self._max_dynamic = max_dynamic_agents
        self._auto_activate = auto_activate_dynamic

    async def build_for(
        self,
        goal: str,
        *,
        task_type: str | None = None,
        requested_capabilities: set[AgentCapability] | None = None,
        privacy: PrivacyLevel = PrivacyLevel.INTERNAL,
    ) -> BuildResult:
        """Build (and register) a dynamic agent definition for the goal."""
        await self._enforce_capacity()

        spec = await self._generate_spec(
            goal,
            task_type=task_type,
            requested_capabilities=requested_capabilities or set(),
            privacy=privacy,
        )
        definition = self._spec_to_definition(spec)
        _validate_dynamic_definition(definition, self._tools)
        existing = await self._catalog.find(definition.agent_id)
        if existing is not None:
            # Reuse an existing matching definition rather than duplicate.
            return BuildResult(
                definition=existing,
                created=False,
                used_llm=spec.metadata.get("used_llm", False)
                if hasattr(spec, "metadata")
                else False,
                metadata={"reused": True},
            )
        await self._catalog.upsert(definition)
        await self._publish(EventType.AGENT_CREATED, definition, extra={"dynamic": True})
        logger.bind(event="agent.dynamic.created", agent_id=definition.agent_id).info(
            "Created dynamic agent '{}' for goal", definition.name
        )
        return BuildResult(
            definition=definition,
            created=True,
            used_llm=bool(spec.used_llm) if hasattr(spec, "used_llm") else False,
        )

    # --- capacity / safety ----------------------------------------------

    async def _enforce_capacity(self) -> None:
        all_defs = await self._catalog.list(limit=10000)
        dynamic_count = sum(1 for d in all_defs if d.dynamic)
        if dynamic_count >= self._max_dynamic:
            raise AgentDefinitionValidationError(f"dynamic agent cap reached ({self._max_dynamic})")

    # --- LLM spec generation --------------------------------------------

    async def _generate_spec(
        self,
        goal: str,
        *,
        task_type: str | None,
        requested_capabilities: set[AgentCapability],
        privacy: PrivacyLevel,
    ) -> _GeneratedSpec:
        if self._llm_enabled and self._router is not None:
            spec = await self._llm_spec(goal, task_type, requested_capabilities, privacy)
            if spec is not None:
                return spec
        return self._fallback_spec(goal, task_type, requested_capabilities, privacy)

    async def _llm_spec(
        self,
        goal: str,
        task_type: str | None,
        requested_capabilities: set[AgentCapability],
        privacy: PrivacyLevel,
    ) -> _GeneratedSpec | None:
        try:
            from app.llm.models import LLMMessage, LLMRequest, MessageRole
        except ImportError:  # pragma: no cover
            return None
        client = await self._select_client()
        if client is None:
            return None
        available_tools = [t.name for t in self._tools.list()]
        messages = [
            LLMMessage(
                role=MessageRole.SYSTEM,
                content=(
                    "You design a new specialized AGENT SPECIFICATION for a task no "
                    "existing agent handles. You produce ONLY a JSON object matching "
                    'this schema (no code, no executable): {"name": str, "purpose": '
                    'str, "description": str, "capabilities": [str], "task_types": '
                    '[str], "required_tools": [str], "preferred_tools": [str], '
                    '"instructions": str, "constraints": [str], '
                    '"verification_strategy": str, "model_requirements": [str], '
                    '"privacy": str (public|internal|private|sensitive), "risk": str '
                    "(low|medium|high|critical)}. Required tools MUST come from the "
                    "available tools list. risk MUST be low or medium for dynamic agents. "
                    "Never request shell/exec/subprocess."
                ),
            ),
            LLMMessage(
                role=MessageRole.USER,
                content=(
                    f"Goal: {goal}\n"
                    f"Task type: {task_type or 'unspecified'}\n"
                    f"Requested capabilities: {sorted(c.value for c in requested_capabilities)}\n"
                    f"Available tools: {available_tools}\n"
                    f"Design an agent specification."
                ),
            ),
        ]
        try:
            request = LLMRequest(
                model="",
                messages=messages,
                temperature=0.2,
                privacy=privacy,
                metadata={"stage": "dynamic_agent_build"},
            )
            response = await client.generate(request)
        except Exception as exc:  # noqa: BLE001 - LLM unavailable → fallback
            logger.bind(event="agent.dynamic.llm.error", error=type(exc).__name__).debug(
                "LLM dynamic build failed (falling back): {}", str(exc)
            )
            return None
        return self._parse_llm_spec(response)

    async def _select_client(self):
        if self._router is None:
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
            metadata={"stage": "dynamic_agent_build"},
        )
        try:
            selection = await self._router.select(request)
        except Exception:  # noqa: BLE001
            return None
        providers = self._router._providers  # noqa: SLF001
        if not providers.exists(selection.provider):
            return None
        return providers.get(selection.provider)

    def _parse_llm_spec(self, response: Any) -> _GeneratedSpec | None:
        import json

        raw = response.content or ""
        try:
            text = raw.strip()
            if text.startswith("```"):
                text = text.strip("`")
                if text.lower().startswith("json"):
                    text = text[4:]
                text = text.strip("`").strip()
            parsed = json.loads(text)
        except Exception:  # noqa: BLE001
            return None
        try:
            caps = {AgentCapability(c) for c in parsed.get("capabilities", []) if c}
            risk = RiskLevel(parsed.get("risk", "low"))
            privacy = PrivacyLevel(parsed.get("privacy", "internal"))
            spec = AgentSpec(
                name=str(parsed.get("name", "Dynamic Agent")).strip(),
                purpose=str(parsed.get("purpose", "")).strip(),
                description=str(parsed.get("description", "")),
                capabilities=set(caps),
                task_types=list(parsed.get("task_types", [])),
                required_tools=list(parsed.get("required_tools", [])),
                preferred_tools=list(parsed.get("preferred_tools", [])),
                instructions=str(parsed.get("instructions", "")),
                constraints=list(parsed.get("constraints", [])),
                verification_strategy=str(parsed.get("verification_strategy", "")),
                model_requirements=list(parsed.get("model_requirements", [])),
                privacy=privacy,
                risk=risk,
                auto_activate=self._auto_activate,
            )
        except (ValueError, TypeError, AgentDefinitionValidationError):
            return None
        return _GeneratedSpec(spec=spec, used_llm=True)

    def _fallback_spec(
        self,
        goal: str,
        task_type: str | None,
        requested_capabilities: set[AgentCapability],
        privacy: PrivacyLevel,
    ) -> _GeneratedSpec:
        """Deterministic spec when the LLM is unavailable.

        Produces a conservative, LOW-risk generic agent that uses only the
        already-registered tools implied by the requested capabilities. It
        never declares CRITICAL/HIGH risk and never requests shell/exec.
        """
        # Pick safe, registered tools matching requested capabilities.
        tools = _safe_tools_for_capabilities(self._tools, requested_capabilities)
        caps = requested_capabilities or {AgentCapability.REASONING}
        spec = AgentSpec(
            name=f"Dynamic {task_type or 'Agent'}",
            purpose=f"Handle tasks of type '{task_type or 'general'}' derived from: {goal[:120]}",
            description="Dynamically generated agent (LLM unavailable; deterministic fallback).",
            capabilities=set(caps),
            task_types=[task_type] if task_type else [],
            required_tools=tools,
            preferred_tools=tools,
            instructions=(
                "Follow the user goal using only registered, policy-gated tools. "
                "Never execute arbitrary code or shell commands."
            ),
            constraints=["no arbitrary code execution", "no shell", "no filesystem bypass"],
            verification_strategy="require successful tool execution evidence",
            model_requirements=["chat", "reasoning"],
            privacy=privacy,
            risk=RiskLevel.LOW,
            auto_activate=self._auto_activate,
        )
        return _GeneratedSpec(spec=spec, used_llm=False)

    # --- spec → definition ----------------------------------------------

    def _spec_to_definition(self, gen: _GeneratedSpec) -> AgentDefinition:
        spec = gen.spec
        # HIGH-risk dynamic agents start INACTIVE (require manual activation);
        # CRITICAL is rejected by validation before reaching here.
        lifecycle = (
            AgentLifecycleState.ACTIVE
            if (self._auto_activate and spec.risk is not RiskLevel.HIGH)
            else AgentLifecycleState.INACTIVE
        )
        agent_id = f"dynamic.{generate_id('agent')[:12]}"
        return AgentDefinition(
            agent_id=agent_id,
            name=spec.name,
            description=spec.description,
            purpose=spec.purpose,
            kind=AgentKind.GENERIC,
            capabilities=set(spec.capabilities),
            task_types=list(spec.task_types),
            required_tools=list(spec.required_tools),
            preferred_tools=list(spec.preferred_tools),
            instructions=spec.instructions,
            constraints=list(spec.constraints),
            verification_strategy=spec.verification_strategy,
            model_requirements=list(spec.model_requirements),
            privacy=spec.privacy,
            risk=spec.risk,
            lifecycle=lifecycle,
            version="0.1.0",
            dynamic=True,
            auto_activate=spec.auto_activate,
            created_at=utc_now(),
            updated_at=utc_now(),
        )

    async def _publish(
        self,
        event_type: EventType,
        definition: AgentDefinition,
        *,
        extra: dict | None = None,
    ) -> None:
        if self._event_bus is None:
            return
        payload = {
            "agent_id": definition.agent_id,
            "name": definition.name,
            "version": definition.version,
            "dynamic": definition.dynamic,
        }
        if extra:
            payload.update(extra)
        await self._event_bus.publish(
            Event.create(
                event_type,
                agent_id=definition.agent_id,
                payload=payload,
                metadata={"agent_id": definition.agent_id, "version": definition.version},
            )
        )


@dataclass(frozen=True)
class _GeneratedSpec:
    spec: AgentSpec
    used_llm: bool


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_FORBIDDEN_TOOL_NAMES = {"shell", "exec", "eval", "subprocess", "python_exec"}
_FORBIDDEN_INSTRUCTION_TOKENS = ("import os\n", "subprocess.", "os.system(", "__import__")


def _validate_dynamic_definition(definition: AgentDefinition, tools: ToolRegistry) -> None:
    """Reject malformed/unsafe/impossible dynamic definitions before registration."""
    if not definition.dynamic:
        return
    if definition.kind is not AgentKind.GENERIC:
        raise AgentDefinitionValidationError(
            "dynamic agents must be GENERIC kind (no code execution surface)"
        )
    if definition.risk is RiskLevel.CRITICAL:
        raise AgentDefinitionValidationError("CRITICAL-risk dynamic agents are never allowed")
    # All required tools must already be registered.
    referenced = set(definition.required_tools) | set(definition.preferred_tools)
    for name in referenced:
        if not tools.exists(name):
            raise AgentDefinitionValidationError(f"required tool '{name}' is not registered")
    bad = _FORBIDDEN_TOOL_NAMES.intersection(referenced)
    if bad:
        raise AgentDefinitionValidationError(
            f"dynamic agent references unrestricted tool(s): {sorted(bad)}"
        )
    # Dynamic agents must never auto-receive HIGH- or CRITICAL-risk tools via an
    # LLM-generated specification. The only way a dynamic agent may run such a
    # tool is a trusted, preconfigured policy pointing at it explicitly — which
    # cannot be introduced by the LLM spec, so we reject references outright.
    high_critical = [
        name
        for name in referenced
        if tools.exists(name)
        and tools.get(name).info.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
    ]
    if high_critical:
        raise AgentDefinitionValidationError(
            f"dynamic agent references HIGH/CRITICAL risk tool(s): {sorted(high_critical)}"
        )
    lowered = definition.instructions.lower()
    if any(tok in lowered for tok in _FORBIDDEN_INSTRUCTION_TOKENS):
        raise AgentDefinitionValidationError(
            "dynamic agent instructions request restricted code execution"
        )
    if not definition.purpose.strip():
        raise AgentDefinitionValidationError("dynamic agent purpose must not be empty")
    if not definition.capabilities:
        raise AgentDefinitionValidationError("dynamic agent must declare at least one capability")


def _safe_tools_for_capabilities(tools: ToolRegistry, caps: set[AgentCapability]) -> list[str]:
    """Pick registered LOW-risk tools whose capabilities match the request."""
    if not caps:
        return []
    names: list[str] = []
    for t in tools.list():
        info = t.info
        if info.risk_level is not RiskLevel.LOW:
            continue
        tool_caps = set(info.capabilities)
        if (
            any(c.value in tool_caps for c in caps)
            or AgentCapability.RESEARCH.value in tool_caps
            and AgentCapability.RESEARCH in caps
            or AgentCapability.MATH.value in tool_caps
            and AgentCapability.MATH in caps
        ):
            names.append(info.name)
    return names
