"""Phase 8.1 — tool foundation.

Covers: tool metadata, capability declaration, registry availability
filtering, executor-time policy enforcement, dynamic-agent HIGH/CRITICAL
restrictions, CRITICAL default denial, configuration limits, structured
ToolResult/Observation compatibility, execution metadata propagation, event
payload safety, and secret-redaction of failure events.

All tests are deterministic and offline. No external APIs are required.
"""

from __future__ import annotations

import pytest

from app.config import ToolSettings, get_settings
from app.events import EventType, InMemoryEventBus
from app.tools.executor import ToolCallRecord, ToolExecutor
from app.tools.impl import CalculatorTool, EchoTool, HealthTool, TimeTool
from app.tools.interface import (
    RiskLevel,
    ToolCategory,
    ToolContext,
    ToolDecisionContext,
    ToolEnvironment,
    ToolExecutionMetadata,
    ToolInfo,
    ToolInterface,
    ToolResult,
)
from app.tools.policy import AllowAllToolPolicy, DefaultToolPolicy
from app.tools.registry import ToolRegistry
from app.tools.sanitize import sanitize_error

# ---------------------------------------------------------------------------
# Tool helpers
# ---------------------------------------------------------------------------


class HighRiskTool(ToolInterface):
    """A HIGH-risk tool for policy/confirmation tests."""

    def __init__(self, name: str = "high_risk") -> None:
        self._info = ToolInfo(
            tool_id=name,
            name=name,
            description="Mock high-risk tool.",
            category=ToolCategory.CUSTOM,
            risk_level=RiskLevel.HIGH,
            input_schema={"type": "object", "properties": {}},
        )

    @property
    def info(self) -> ToolInfo:
        return self._info

    async def execute(self, context: ToolContext) -> ToolResult:
        return ToolResult(
            invocation_id=context.invocation_id,
            tool_call_id=context.tool_call_id,
            name=self.name,
            output="high",
        )


class CriticalTool(ToolInterface):
    def __init__(self, name: str = "critical_tool") -> None:
        self._info = ToolInfo(
            tool_id=name,
            name=name,
            description="Mock critical tool.",
            risk_level=RiskLevel.CRITICAL,
        )

    @property
    def info(self) -> ToolInfo:
        return self._info

    async def execute(self, context: ToolContext) -> ToolResult:
        return ToolResult(invocation_id=context.invocation_id, name=self.name, output="critical")


class SecretsTool(ToolInterface):
    """A tool that always fails with an exception embedding a secret."""

    def __init__(self) -> None:
        self._info = ToolInfo(name="secrets")

    @property
    def info(self) -> ToolInfo:
        return self._info

    async def execute(self, context: ToolContext) -> ToolResult:
        raise ValueError(
            "api_key=sk-super-secret-12345 denied; Authorization: Bearer abcd12345efgh6789"
        )


class PartialTool(ToolInterface):
    """A tool returning a partial result with structured evidence."""

    def __init__(self) -> None:
        self._info = ToolInfo(name="partial_tool")

    @property
    def info(self) -> ToolInfo:
        return self._info

    async def execute(self, context: ToolContext) -> ToolResult:
        return ToolResult(
            invocation_id=context.invocation_id,
            name=self.name,
            success=True,
            partial=True,
            output={"items_fetched": 10, "items_total": 100},
            evidence={"page": 1, "truncated": True},
        )


async def _register_defaults(reg: ToolRegistry) -> None:
    for tool in [CalculatorTool(), TimeTool(), HealthTool(), EchoTool()]:
        await reg.register(tool)


# ---------------------------------------------------------------------------
# 1. Tool metadata
# ---------------------------------------------------------------------------


def test_tool_metadata_fields_exist() -> None:
    info = ToolInfo(
        name="fs.read",
        description="Read a file.",
        category=ToolCategory.FILESYSTEM,
        version="2.0.0",
        risk_level=RiskLevel.MEDIUM,
        read_only=True,
        modifies_external_state=False,
        supported_environments=[ToolEnvironment.CONTAINER, ToolEnvironment.LOCAL],
        default_timeout_seconds=12.5,
        required_capabilities=["filesystem:read"],
    )
    assert info.name == "fs.read"
    assert info.description == "Read a file."
    assert info.category is ToolCategory.FILESYSTEM
    assert info.version == "2.0.0"
    assert info.read_only is True
    assert info.modifies_external_state is False
    assert info.default_timeout_seconds == 12.5
    assert info.required_capabilities == ["filesystem:read"]
    assert info.supported_environments == [
        ToolEnvironment.CONTAINER,
        ToolEnvironment.LOCAL,
    ]


def test_tool_info_defaults() -> None:
    info = ToolInfo(name="x", risk_level=RiskLevel.LOW)
    # read_only/modifies_external_state default to None (unknown) — not assumed.
    assert info.read_only is None
    assert info.modifies_external_state is None
    assert info.supported_environments == [ToolEnvironment.LOCAL]
    assert info.default_timeout_seconds is None
    assert info.read_only is None


def test_existing_tool_info_still_constructs() -> None:
    # Regression: Phase 0-7 ToolInfo constructions must keep working.
    info = CalculatorTool().info
    assert info.name == "calculator"
    assert info.risk_level is RiskLevel.LOW
    assert info.capabilities == ["arithmetic"]
    assert info.supported_environments == [ToolEnvironment.LOCAL]


# ---------------------------------------------------------------------------
# 2. Capability declaration
# ---------------------------------------------------------------------------


def test_effective_capabilities_union() -> None:
    info = ToolInfo(
        name="act",
        capabilities=["a", "b"],
        required_capabilities=["x", "y"],
    )
    # Both lists are declarative; required_capabilities are the *needs*.
    assert sorted(info.required_capabilities) == ["x", "y"]
    assert sorted(info.capabilities) == ["a", "b"]


def test_decision_context_carries_agent_and_task_identity() -> None:
    ctx = ToolDecisionContext(
        agent_id="a1",
        agent_version="1.2.3",
        agent_dynamic=True,
        task_id="t1",
        session_id="s1",
        operation="discover",
        environment=ToolEnvironment.CONTAINER,
    )
    assert ctx.agent_id == "a1"
    assert ctx.agent_version == "1.2.3"
    assert ctx.agent_dynamic is True
    assert ctx.environment is ToolEnvironment.CONTAINER


# ---------------------------------------------------------------------------
# 3. Registry availability filtering (discovery)
# ---------------------------------------------------------------------------


async def test_list_available_filters_by_policy() -> None:
    reg = ToolRegistry()
    await reg.register(CalculatorTool())
    await reg.register(HighRiskTool())
    await reg.register(CriticalTool())

    policy = DefaultToolPolicy()  # critical denied, high requires confirmation
    avail = await reg.list_available(policy=policy)
    names = {i.name for i in avail}
    # LOW/allowable tools available; CRITICAL not available.
    assert "calculator" in names
    assert "critical_tool" not in names
    # HIGH requires confirmation => not "allowed" for discovery.
    assert "high_risk" not in names


async def test_list_available_allow_all_includes_all() -> None:
    reg = ToolRegistry()
    await reg.register(CalculatorTool())
    await reg.register(HighRiskTool())
    avail = await reg.list_available(policy=AllowAllToolPolicy())
    assert {i.name for i in avail} == {"calculator", "high_risk"}


async def test_list_available_environments_filter() -> None:
    reg = ToolRegistry()
    local = ToolInfo(name="local_tool", supported_environments=[ToolEnvironment.LOCAL])
    remote = ToolInfo(name="remote_tool", supported_environments=[ToolEnvironment.REMOTE])
    await reg.register(_StaticTool(local))
    await reg.register(_StaticTool(remote))
    avail = await reg.list_available(
        policy=AllowAllToolPolicy(), environments={ToolEnvironment.REMOTE}
    )
    assert {i.name for i in avail} == {"remote_tool"}


# ---------------------------------------------------------------------------
# 4. Executor-time policy enforcement
# ---------------------------------------------------------------------------


async def test_executor_denies_critical_at_authorization() -> None:
    reg = ToolRegistry()
    await reg.register(CriticalTool())
    policy = DefaultToolPolicy()
    ex = ToolExecutor(registry=reg, policy=policy)
    outcome = await ex.execute_call(
        ToolCallRecord(tool_call_id="c1", tool_name="critical_tool", arguments={})
    )
    assert outcome.skipped is True
    assert outcome.decision is not None and outcome.decision.denied
    assert "CRITICAL" in outcome.decision.reason


async def test_discovery_is_not_a_security_boundary() -> None:
    # Listing availability must not bypass executor enforcement: even if a
    # caller "sees" a tool via AllowAllPolicy discovery, DefaultToolPolicy
    # still denies CRITICAL at execution time.
    reg = ToolRegistry()
    await reg.register(CriticalTool())
    # Discovery via default policy hides CRITICAL…
    assert "critical_tool" not in {
        i.name for i in await reg.list_available(policy=DefaultToolPolicy())
    }
    # …but executor enforcement is independent and authoritative.
    ex = ToolExecutor(registry=reg, policy=DefaultToolPolicy())
    outcome = await ex.execute_call(
        ToolCallRecord(tool_call_id="c2", tool_name="critical_tool", arguments={})
    )
    assert outcome.skipped is True


async def test_executor_allows_explicit_critical_allow() -> None:
    reg = ToolRegistry()
    await reg.register(CriticalTool())
    policy = DefaultToolPolicy(allow={"critical_tool"})
    ex = ToolExecutor(registry=reg, policy=policy)
    outcome = await ex.execute_call(
        ToolCallRecord(tool_call_id="c3", tool_name="critical_tool", arguments={})
    )
    assert outcome.result is not None and outcome.result.success
    assert outcome.result.output == "critical"


# ---------------------------------------------------------------------------
# 5. Dynamic-agent HIGH/CRITICAL restrictions
# ---------------------------------------------------------------------------


def _dynamic_context(agent_id: str = "dynamic.abc") -> ToolDecisionContext:
    return ToolDecisionContext(agent_id=agent_id, agent_dynamic=True)


async def test_dynamic_agent_discovery_excludes_high_and_critical() -> None:
    reg = ToolRegistry()
    await reg.register(CalculatorTool())  # LOW
    await reg.register(HighRiskTool())
    await reg.register(CriticalTool())
    avail = await reg.list_available(policy=AllowAllToolPolicy(), context=_dynamic_context())
    names = {i.name for i in avail}
    assert "calculator" in names
    assert "high_risk" not in names
    assert "critical_tool" not in names


async def test_dynamic_agent_executor_still_enforces_high_confirm() -> None:
    # Even if a dynamic agent saw HIGH tool via a permissive discovery policy,
    # the executor policy still requires confirmation and never auto-executes.
    reg = ToolRegistry()
    await reg.register(HighRiskTool())
    # A somewhat-permissive policy that auto-approves HIGH (trusted preconfig)
    policy = DefaultToolPolicy(auto_approve={"high_risk"})
    ex = ToolExecutor(registry=reg, policy=policy)
    outcome = await ex.execute_call(
        ToolCallRecord(tool_call_id="c4", tool_name="high_risk", arguments={})
    )
    assert outcome.result is not None and outcome.result.success
    # Explicit allow came from preconfigured *policy*, not the dynamic agent.
    assert outcome.decision is not None
    assert outcome.decision.metadata.get("explicit") is True


async def test_dynamic_agent_critical_always_denied_by_default() -> None:
    # CRITICAL denied by default even for dynamic agents; no auto-approval based
    # on agent metadata. Approval requires an explicit trusted policy allow.
    reg = ToolRegistry()
    await reg.register(CriticalTool())
    ex = ToolExecutor(registry=reg, policy=DefaultToolPolicy())
    outcome = await ex.execute_call(
        ToolCallRecord(tool_call_id="c5", tool_name="critical_tool", arguments={}),
        context=_dynamic_context(),
    )
    assert outcome.skipped is True
    assert outcome.decision is not None and outcome.decision.denied


# ---------------------------------------------------------------------------
# 6. CRITICAL default denial
# ---------------------------------------------------------------------------


def test_critical_default_denied_no_auto_approve() -> None:
    policy = DefaultToolPolicy()
    decision = policy.evaluate(CriticalTool().info)
    assert decision.denied
    assert policy.evaluate(CriticalTool().info).denied


def test_no_automatic_critical_approval_in_policy() -> None:
    # There is intentionally no mechanism that auto-approves CRITICAL purely
    # from tool metadata. A CRITICAL tool requires an explicit allow entry.
    policy = DefaultToolPolicy()
    assert policy.evaluate(CriticalTool().info).denied
    allowed = DefaultToolPolicy(allow={"critical_tool"})
    assert allowed.evaluate(CriticalTool().info).allowed


# ---------------------------------------------------------------------------
# 12. Defect regressions: CRITICAL/auto_approve + dynamic-agent execution
# ---------------------------------------------------------------------------


def test_auto_approve_never_allows_critical() -> None:
    # Defect 1 regression: auto_approve must NEVER convert a CRITICAL deny into
    # an ALLOW, even when the tool name is present in auto_approve.
    policy = DefaultToolPolicy(auto_approve={"critical_tool"})
    decision = policy.evaluate(CriticalTool().info)
    assert decision.denied
    assert "CRITICAL" in decision.reason


async def test_executor_auto_approve_never_allows_critical() -> None:
    reg = ToolRegistry()
    await reg.register(CriticalTool())
    ex = ToolExecutor(registry=reg, policy=DefaultToolPolicy(auto_approve={"critical_tool"}))
    outcome = await ex.execute_call(
        ToolCallRecord(tool_call_id="c12", tool_name="critical_tool", arguments={})
    )
    assert outcome.skipped is True
    assert outcome.decision is not None and outcome.decision.denied


async def test_dynamic_agent_cannot_execute_high_through_real_executor() -> None:
    # Defect 2 regression: under the default (risk-aware) runtime policy a
    # dynamic agent executing a HIGH-risk tool through the real executor path
    # (dynamic identity propagated) is DENIED outright — not left pending
    # confirmation and never auto-approved.
    reg = ToolRegistry()
    await reg.register(HighRiskTool())
    ex = ToolExecutor(registry=reg, policy=DefaultToolPolicy())
    outcome = await ex.execute_call(
        ToolCallRecord(tool_call_id="c13", tool_name="high_risk", arguments={}),
        context=_dynamic_context(),
    )
    assert outcome.skipped is True
    assert outcome.decision is not None and outcome.decision.denied
    assert not outcome.decision.requires_confirmation


async def test_dynamic_agent_high_permitted_by_trusted_auto_approve() -> None:
    # The requirement allows a trusted, preconfigured policy to explicitly
    # permit a HIGH tool for a dynamic agent. auto_approve is such a trusted
    # preconfig, so the dynamic deny must NOT override it.
    reg = ToolRegistry()
    await reg.register(HighRiskTool())
    ex = ToolExecutor(registry=reg, policy=DefaultToolPolicy(auto_approve={"high_risk"}))
    outcome = await ex.execute_call(
        ToolCallRecord(tool_call_id="c15", tool_name="high_risk", arguments={}),
        context=_dynamic_context(),
    )
    assert outcome.result is not None and outcome.result.success
    assert outcome.decision is not None and outcome.decision.allowed


async def test_dynamic_agent_cannot_execute_critical_through_real_executor() -> None:
    # Defect 2 regression: dynamic agent executing CRITICAL via the executor is
    # denied even if the tool appears in auto_approve (trusted allow list is the
    # ONLY escape, and it must be preconfigured, not dynamic-agent supplied).
    reg = ToolRegistry()
    await reg.register(CriticalTool())
    ex = ToolExecutor(registry=reg, policy=DefaultToolPolicy(auto_approve={"critical_tool"}))
    outcome = await ex.execute_call(
        ToolCallRecord(tool_call_id="c14", tool_name="critical_tool", arguments={}),
        context=_dynamic_context(),
    )
    assert outcome.skipped is True
    assert outcome.decision is not None and outcome.decision.denied
    assert "CRITICAL" in outcome.decision.reason


def test_dynamic_agent_non_dynamic_high_still_confirms() -> None:
    # Sanity: for a non-dynamic (static) agent a HIGH tool still requires
    # confirmation (unchanged Phase 0-7 exposure) rather than outright denial.
    policy = DefaultToolPolicy()
    decision = policy.evaluate(
        HighRiskTool().info, context=ToolDecisionContext(agent_dynamic=False)
    )
    assert decision.requires_confirmation
    assert not decision.denied


async def test_llm_dynamic_spec_cannot_request_high_or_critical_tools() -> None:
    # Defect 2/B: an LLM-generated dynamic agent specification must be rejected
    # if it references a HIGH or CRITICAL registered tool. The spec (untrusted)
    # must not be able to pull a risky tool into its required/preferred list.
    from app.agents.definitions import AgentDefinition, AgentKind
    from app.agents.dynamic_builder import _validate_dynamic_definition
    from app.agents.interface import AgentCapability
    from app.core.exceptions import AgentDefinitionValidationError

    tool_reg = ToolRegistry()
    await tool_reg.register(HighRiskTool())
    await tool_reg.register(CriticalTool())

    risky = AgentDefinition(
        agent_id="dyn.risky",
        name="Risky",
        kind=AgentKind.GENERIC,
        capabilities={AgentCapability.REASONING},
        dynamic=True,
        risk=RiskLevel.LOW,
        required_tools=["high_risk"],
    )
    with pytest.raises(AgentDefinitionValidationError):
        _validate_dynamic_definition(risky, tool_reg)

    critical = AgentDefinition(
        agent_id="dyn.crit",
        name="Crit",
        kind=AgentKind.GENERIC,
        capabilities={AgentCapability.REASONING},
        dynamic=True,
        risk=RiskLevel.LOW,
        required_tools=["critical_tool"],
    )
    with pytest.raises(AgentDefinitionValidationError):
        _validate_dynamic_definition(critical, tool_reg)


async def test_dynamic_generic_agent_denied_high_via_real_executor() -> None:
    # End-to-end production path: a dynamic GenericAgent runs its required
    # tools through the shared ToolExecutor. Because GenericAgent propagates
    # agent_dynamic into the decision context, a HIGH tool is DENIED (not
    # merely pending confirmation), and the agent records the failure without
    # the HIGH tool ever executing.
    from app.agents.definitions import AgentDefinition, AgentKind
    from app.agents.factory import GenericAgent
    from app.agents.interface import AgentCapability, AgentContext, AgentStatus

    reg = ToolRegistry()
    await reg.register(HighRiskTool())
    # Default (risk-aware) runtime policy: a HIGH tool is not pre-authorized
    # anywhere, so the dynamic agent must be denied at the executor boundary.
    ex = ToolExecutor(registry=reg, policy=DefaultToolPolicy())
    definition = AgentDefinition(
        agent_id="dynamic.gen",
        name="DynamicGen",
        kind=AgentKind.GENERIC,
        capabilities={AgentCapability.REASONING},
        dynamic=True,
        risk=RiskLevel.LOW,
        required_tools=["high_risk"],
    )
    agent = GenericAgent(definition, tool_executor=ex)
    result = await agent.execute(AgentContext(task_id="t-dyn", input="do it"))
    assert result.status is AgentStatus.FAILED
    assert "high_risk" in {obs["tool"] for obs in result.output["observations"]}
    failing = [o for o in result.output["observations"] if o["tool"] == "high_risk"][0]
    # The HIGH tool never executed -> denied, and the agent surfaces a failure.
    assert failing["success"] is False
    assert not any(o.get("output") == "high" for o in result.output["observations"])


# ---------------------------------------------------------------------------
# 7. Configuration limits (single source of truth)
# ---------------------------------------------------------------------------


def test_tool_settings_defaults() -> None:
    s = ToolSettings()
    assert s.max_execution_time_seconds == 60.0
    assert s.max_output_size_bytes == 100_000
    assert s.max_redirects == 5
    assert s.max_downloaded_bytes == 10_000_000
    assert s.max_filesystem_file_bytes == 1_000_000


def test_tool_settings_aggregated_in_get_settings() -> None:
    settings = get_settings()
    assert isinstance(settings.tools, ToolSettings)


def test_tool_settings_rejects_invalid(monkeypatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("TOOL_MAX_EXECUTION_TIME_SECONDS", "0")
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        ToolSettings()
    get_settings.cache_clear()


def test_executor_exposes_single_limit_source() -> None:
    ex = ToolExecutor(registry=ToolRegistry(), policy=AllowAllToolPolicy())
    effective = ex.effective_limits
    assert isinstance(effective, ToolSettings)
    # Repeated access yields an equivalent single source of truth.
    assert ex.effective_limits.max_execution_time_seconds == effective.max_execution_time_seconds


def test_tool_default_timeout_preferred_over_limit() -> None:
    ex = ToolExecutor(registry=ToolRegistry(), policy=AllowAllToolPolicy())
    info = ToolInfo(name="t", default_timeout_seconds=15.0)
    assert ex._execution_timeout(info, None) == 15.0
    assert ex._execution_timeout(info, 9.0) == 9.0


def test_tool_timeout_has_single_authoritative_source() -> None:
    # Defect 4 regression: the per-tool execution timeout must resolve from the
    # executor's single authoritative source (ToolSettings.max_execution_time_seconds),
    # and must NOT be a separately-configurable duplicate on the loop policy.
    ex = ToolExecutor(registry=ToolRegistry(), policy=DefaultToolPolicy())
    configured = get_settings().tools.max_execution_time_seconds
    # No explicit timeout, no tool default -> the ToolSettings value is used.
    assert ex._execution_timeout(ToolInfo(name="t"), None) == configured
    # LoopPolicy carries its per-step budget, but the tool-executor timeout is
    # not derived from it (so the two cannot silently diverge).
    from app.runtime.loop.loop_policy import LoopPolicy

    loop_step_budget = LoopPolicy().per_execution_timeout_seconds
    # With the default configuration both happen to be 60.0, but the executor
    # resolves the tool timeout from ToolSettings, not from LoopPolicy. Changing
    # only ToolSettings must change the executor resolution.
    override = ToolSettings(
        max_execution_time_seconds=configured + 7.0,
        max_output_size_bytes=100_000,
        max_redirects=5,
        max_downloaded_bytes=10_000_000,
        max_filesystem_file_bytes=1_000_000,
    )
    ex2 = ToolExecutor(registry=ToolRegistry(), policy=DefaultToolPolicy(), limits=override)
    assert ex2._execution_timeout(ToolInfo(name="t"), None) == configured + 7.0
    assert loop_step_budget is not None


async def test_output_size_limit_enforced() -> None:
    reg = ToolRegistry()
    big = _StaticTool(ToolInfo(name="big"), output={"large": "x" * 1000})
    await reg.register(big)
    # A full instance (single source of truth) with a tiny output cap.
    limits = ToolSettings(
        max_execution_time_seconds=60.0,
        max_output_size_bytes=8,
        max_redirects=5,
        max_downloaded_bytes=10_000_000,
        max_filesystem_file_bytes=1_000_000,
    )
    ex = ToolExecutor(registry=reg, policy=AllowAllToolPolicy(), limits=limits)
    outcome = await ex.execute_call(
        ToolCallRecord(tool_call_id="c6", tool_name="big", arguments={})
    )
    assert outcome.result is not None
    assert outcome.result.success is False
    assert outcome.result.partial is True
    assert outcome.result.error_type == "output_limit_exceeded"
    # Defect 3: the oversized raw output must NOT remain available in the
    # returned result, the observation, or the agent-facing output.
    assert outcome.result.output == {"truncated": True}
    assert "x" * 1000 not in repr(outcome.result.output)
    assert outcome.observation.content != {"large": "x" * 1000}


async def test_output_limit_never_emits_oversized_payload_to_observation() -> None:
    # Simulates what a caller (e.g. GenericAgent) reads from the outcome: the
    # full oversized payload must not survive anywhere reachable from the
    # returned ToolResult/observation.
    reg = ToolRegistry()
    big = _StaticTool(ToolInfo(name="big"), output={"secret_large": "S" * 5000})
    await reg.register(big)
    limits = ToolSettings(
        max_execution_time_seconds=60.0,
        max_output_size_bytes=16,
        max_redirects=5,
        max_downloaded_bytes=10_000_000,
        max_filesystem_file_bytes=1_000_000,
    )
    ex = ToolExecutor(registry=reg, policy=AllowAllToolPolicy(), limits=limits)
    outcome = await ex.execute_call(
        ToolCallRecord(tool_call_id="c11", tool_name="big", arguments={})
    )
    assert outcome.result is not None
    assert outcome.result.success is False
    # The oversized payload must not survive anywhere reachable from the outcome:
    # not in the observation content (Observation is a pydantic model, exposed via
    # .content/.model_dump()), not in the ToolResult.output, and not in a
    # stringified form of the plain-dataclass outcome.
    assert outcome.observation.content != {"secret_large": "S" * 5000}
    dumped_obs = outcome.observation.model_dump()
    assert "S" * 5000 not in str(dumped_obs) and "S" * 5000 not in repr(dumped_obs)
    assert "S" * 5000 not in repr(outcome)
    # The only raw-output bearer (ToolResult.output) is the bounded marker.
    assert outcome.result.output == {"truncated": True}


# ---------------------------------------------------------------------------
# 8. Structured ToolResult / Observation compatibility
# ---------------------------------------------------------------------------


def test_partial_result_and_evidence() -> None:
    req = partial_result()
    assert req.success is True
    assert req.partial is True
    assert req.evidence["truncated"] is True


async def test_partial_tool_result_runs_through_executor() -> None:
    reg = ToolRegistry()
    await reg.register(PartialTool())
    ex = ToolExecutor(registry=reg, policy=AllowAllToolPolicy())
    outcome = await ex.execute_call(
        ToolCallRecord(tool_call_id="c7", tool_name="partial_tool", arguments={})
    )
    assert outcome.result is not None
    assert outcome.result.partial is True
    assert outcome.result.evidence["truncated"] is True
    # Observation remains compatible (safe normalized content).
    assert outcome.observation.success is True


# ---------------------------------------------------------------------------
# 9. Execution metadata propagation
# ---------------------------------------------------------------------------


async def test_execution_metadata_propagates() -> None:
    reg = ToolRegistry()
    await _register_defaults(reg)
    ex = ToolExecutor(registry=reg, policy=DefaultToolPolicy())
    outcome = await ex.execute_call(
        ToolCallRecord(tool_call_id="c8", tool_name="time", arguments={}),
        task_id="t1",
        session_id="s1",
        iteration=3,
        loop_id="loop1",
        context=ToolDecisionContext(
            agent_id="agent1", agent_version="1.0.0", task_id="t1", session_id="s1"
        ),
    )
    meta = outcome.result.execution_metadata
    assert isinstance(meta, ToolExecutionMetadata)
    assert meta.task_id == "t1"
    assert meta.session_id == "s1"
    assert meta.iteration == 3
    assert meta.loop_id == "loop1"
    assert meta.agent_id == "agent1"
    assert meta.agent_version == "1.0.0"
    assert meta.policy_verdict == "allow"
    assert meta.success is True
    assert meta.duration_ms is not None


def test_execution_metadata_frozen_and_forbids_extra() -> None:
    import pydantic

    try:
        ToolExecutionMetadata(tool="x", extra_field="nope")  # type: ignore[call-arg]
        raise AssertionError("extra fields must be forbidden")
    except pydantic.ValidationError:
        pass


# ---------------------------------------------------------------------------
# 10. Event payload safety
# ---------------------------------------------------------------------------


async def test_tool_events_carry_safe_payload() -> None:
    reg = ToolRegistry()
    await _register_defaults(reg)
    bus = InMemoryEventBus()
    events: list = []
    bus.subscribe(None, lambda ev: events.append(ev))
    ex = ToolExecutor(registry=reg, policy=DefaultToolPolicy(), event_bus=bus)
    await ex.execute_call(
        ToolCallRecord(tool_call_id="c9", tool_name="echo", arguments={"x": 1}),
        task_id="t1",
    )
    types = {e.event_type for e in events}
    assert EventType.TOOL_CALL_REQUESTED in types
    assert EventType.TOOL_CALL_STARTED in types
    assert EventType.TOOL_CALL_COMPLETED in types

    # Discovery event payload is metadata-only.
    reg2 = ToolRegistry(event_bus=bus)
    await _register_defaults(reg2)
    events.clear()
    await reg2.list_available(policy=DefaultToolPolicy(), context=ToolDecisionContext(task_id="tX"))
    disc = [e for e in events if e.event_type is EventType.TOOL_DISCOVERED]
    assert disc
    for ev in disc:
        joined = f"{ev.payload} {ev.metadata}"
        assert "sk-secret" not in joined
        assert "Bearer" not in joined
        assert "password" not in joined.lower()


# ---------------------------------------------------------------------------
# 11. Secret-containing exception does not leak into failure events
# ---------------------------------------------------------------------------


async def test_failure_event_redacts_secret() -> None:
    reg = ToolRegistry()
    await reg.register(SecretsTool())
    bus = InMemoryEventBus()
    events: list = []
    bus.subscribe(None, lambda ev: events.append(ev))
    ex = ToolExecutor(registry=reg, policy=DefaultToolPolicy(), event_bus=bus)
    outcome = await ex.execute_call(
        ToolCallRecord(tool_call_id="c10", tool_name="secrets", arguments={})
    )
    assert outcome.result is not None and not outcome.result.success
    # The ToolResult error itself is sanitized.
    assert "sk-super-secret-12345" not in outcome.result.error
    assert "abcd12345efgh6789" not in outcome.result.error
    # Failure events carry only sanitized error.
    fail_events = [e for e in events if e.event_type is EventType.TOOL_CALL_FAILED]
    assert fail_events
    for ev in fail_events:
        joined = f"{ev.payload} {ev.metadata}"
        assert "sk-super-secret-12345" not in joined
        assert "abcd12345efgh6789" not in joined


def test_sanitize_error_redacts_variants() -> None:
    assert sanitize_error("failed api_key=abc123") == "failed [redacted]"
    assert sanitize_error("password secret") == "password secret"  # bare word untouched
    # Security property: the bearer token value is always removed, regardless of
    # the exact placeholder string used.
    bearer_sanitized = sanitize_error("Bearer abcdef1234567890abcdef1234567890")
    assert "abcdef1234567890" not in bearer_sanitized
    assert bearer_sanitized == "Bearer [redacted]"  # existing redaction convention
    assert "abcdef1234567890" not in sanitize_error("token=abcdef1234567890")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


class _StaticTool(ToolInterface):
    """A tool backed by a pre-built ToolInfo (for metadata tests)."""

    def __init__(self, info: ToolInfo, output: object = "ok") -> None:
        self._info = info
        self._output = output

    @property
    def info(self) -> ToolInfo:
        return self._info

    async def execute(self, context: ToolContext) -> ToolResult:
        return ToolResult(invocation_id=context.invocation_id, name=self.name, output=self._output)


def partial_result() -> ToolResult:
    return ToolResult(
        invocation_id="i1",
        name="partial_tool",
        success=True,
        partial=True,
        output={"items_fetched": 10, "items_total": 100},
        evidence={"page": 1, "truncated": True},
        execution_metadata=ToolExecutionMetadata(tool="partial_tool", success=True),
    )
