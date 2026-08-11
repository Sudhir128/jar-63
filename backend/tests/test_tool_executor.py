"""Tests for the ToolExecutor, ToolPolicy, and confirmation flow."""

from __future__ import annotations

import pytest

from app.events import EventType, InMemoryEventBus
from app.tools.confirmation import ConfirmationStore
from app.tools.executor import ToolCallRecord, ToolExecutor
from app.tools.impl import CalculatorTool, EchoTool, HealthTool, TimeTool
from app.tools.interface import (
    RiskLevel,
    ToolCategory,
    ToolContext,
    ToolInfo,
    ToolInterface,
    ToolResult,
)
from app.tools.policy import (
    AllowAllToolPolicy,
    DefaultToolPolicy,
    PolicyDecision,
)
from app.tools.registry import ToolRegistry


class MockHighRiskTool(ToolInterface):
    """A mock HIGH-risk tool for policy/confirmation tests."""

    def __init__(self, name: str = "mock_high") -> None:
        self._info = ToolInfo(
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
            success=True,
            output="high risk result",
        )


class MockCriticalTool(ToolInterface):
    def __init__(self, name: str = "mock_critical") -> None:
        self._info = ToolInfo(
            name=name, description="Mock critical tool.", risk_level=RiskLevel.CRITICAL
        )

    @property
    def info(self) -> ToolInfo:
        return self._info

    async def execute(self, context: ToolContext) -> ToolResult:
        return ToolResult(invocation_id=context.invocation_id, name=self.name, output="critical")


async def _register_defaults(reg: ToolRegistry) -> None:
    for t in [CalculatorTool(), TimeTool(), HealthTool(), EchoTool()]:
        await reg.register(t)


# --- ToolPolicy ---


def test_allow_all_policy_allows_everything() -> None:
    policy = AllowAllToolPolicy()
    decision = policy.evaluate(CalculatorTool().info)
    assert decision.allowed


def test_default_policy_allows_low_risk() -> None:
    policy = DefaultToolPolicy()
    decision = policy.evaluate(CalculatorTool().info)
    assert decision.allowed


def test_default_policy_denies_explicit_deny() -> None:
    policy = DefaultToolPolicy(deny={"calculator"})
    decision = policy.evaluate(CalculatorTool().info)
    assert decision.denied


def test_default_policy_requires_confirmation_for_high() -> None:
    policy = DefaultToolPolicy()
    decision = policy.evaluate(MockHighRiskTool().info)
    assert decision.requires_confirmation


def test_default_policy_denies_critical_unless_allowed() -> None:
    policy = DefaultToolPolicy()
    decision = policy.evaluate(MockCriticalTool().info)
    assert decision.denied
    assert "CRITICAL" in decision.reason


def test_default_policy_allows_critical_when_explicitly_allowed() -> None:
    policy = DefaultToolPolicy(allow={"mock_critical"})
    decision = policy.evaluate(MockCriticalTool().info)
    assert decision.allowed


def test_default_policy_auto_approve_overrides_confirmation() -> None:
    policy = DefaultToolPolicy(auto_approve={"mock_high"})
    decision = policy.evaluate(MockHighRiskTool().info)
    assert decision.allowed


def test_default_policy_denies_network_when_disabled() -> None:
    tool = type(
        "NetTool",
        (ToolInterface,),
        {
            "_info": ToolInfo(name="net_tool", requires_network=True, risk_level=RiskLevel.LOW),
            "info": property(lambda self: self._info),
            "execute": lambda self, ctx: ToolResult(
                invocation_id=ctx.invocation_id, name="net_tool"
            ),
        },
    )()
    policy = DefaultToolPolicy(network_allowed=False)
    decision = policy.evaluate(tool.info)
    assert decision.denied
    assert "network" in decision.reason.lower()


def test_policy_decision_helpers() -> None:
    allow = PolicyDecision.allow("ok")
    assert allow.allowed and not allow.denied
    deny = PolicyDecision.deny("no")
    assert deny.denied and not deny.allowed
    confirm = PolicyDecision.confirm("ask")
    assert confirm.requires_confirmation and not confirm.allowed


# --- ToolExecutor: resolution & validation ---


async def test_executor_resolves_registered_tool() -> None:
    reg = ToolRegistry()
    await _register_defaults(reg)
    ex = ToolExecutor(registry=reg, policy=DefaultToolPolicy())
    call = ToolCallRecord(
        tool_call_id="c1", tool_name="calculator", arguments={"expression": "2+2"}
    )
    outcome = await ex.execute_call(call, task_id="t1")
    assert outcome.result is not None
    assert outcome.result.success
    assert outcome.result.output["result"] == 4


async def test_executor_rejects_unregistered_tool() -> None:
    reg = ToolRegistry()
    await _register_defaults(reg)
    ex = ToolExecutor(registry=reg, policy=DefaultToolPolicy())
    call = ToolCallRecord(tool_call_id="c1", tool_name="nonexistent", arguments={})
    outcome = await ex.execute_call(call, task_id="t1")
    assert outcome.skipped
    assert outcome.result is None
    assert "not registered" in outcome.skipped_reason


async def test_executor_validates_arguments_schema() -> None:
    reg = ToolRegistry()
    await _register_defaults(reg)
    ex = ToolExecutor(registry=reg, policy=DefaultToolPolicy())
    # Missing required 'expression' field.
    call = ToolCallRecord(tool_call_id="c1", tool_name="calculator", arguments={})
    outcome = await ex.execute_call(call, task_id="t1")
    assert outcome.skipped
    assert "Invalid arguments" in outcome.skipped_reason


async def test_executor_normalizes_tool_failure() -> None:
    reg = ToolRegistry()
    await _register_defaults(reg)
    ex = ToolExecutor(registry=reg, policy=DefaultToolPolicy())
    call = ToolCallRecord(
        tool_call_id="c1", tool_name="calculator", arguments={"expression": "1/0"}
    )
    outcome = await ex.execute_call(call, task_id="t1")
    assert outcome.result is not None
    assert not outcome.result.success
    assert "Division by zero" in outcome.result.error
    assert outcome.observation.success is False


async def test_executor_creates_observation_on_success() -> None:
    reg = ToolRegistry()
    await _register_defaults(reg)
    ex = ToolExecutor(registry=reg, policy=DefaultToolPolicy())
    call = ToolCallRecord(tool_call_id="c1", tool_name="time", arguments={})
    outcome = await ex.execute_call(call, task_id="t1")
    assert outcome.observation.success
    assert outcome.observation.tool_name == "time"
    assert outcome.observation.tool_call_id == "c1"


# --- ToolExecutor: policy enforcement ---


async def test_executor_denies_tool_via_policy() -> None:
    reg = ToolRegistry()
    await _register_defaults(reg)
    ex = ToolExecutor(registry=reg, policy=DefaultToolPolicy(deny={"calculator"}))
    call = ToolCallRecord(
        tool_call_id="c1", tool_name="calculator", arguments={"expression": "1+1"}
    )
    outcome = await ex.execute_call(call, task_id="t1")
    assert outcome.skipped
    assert outcome.decision is not None
    assert outcome.decision.denied
    assert outcome.observation.type.value == "tool_denied"


async def test_executor_requires_confirmation_for_high_risk() -> None:
    reg = ToolRegistry()
    await reg.register(MockHighRiskTool())
    store = ConfirmationStore()
    ex = ToolExecutor(registry=reg, policy=DefaultToolPolicy(), confirmation_store=store)
    call = ToolCallRecord(tool_call_id="c1", tool_name="mock_high", arguments={})
    outcome = await ex.execute_call(call, task_id="t1")
    assert outcome.skipped
    assert outcome.confirmation is not None
    assert outcome.confirmation.is_pending
    assert outcome.observation.type.value == "tool_confirmation_required"
    assert outcome.result is None  # tool was NOT executed


async def test_executor_executes_after_confirmation_approved() -> None:
    reg = ToolRegistry()
    await reg.register(MockHighRiskTool())
    store = ConfirmationStore()
    ex = ToolExecutor(registry=reg, policy=DefaultToolPolicy(), confirmation_store=store)
    call = ToolCallRecord(tool_call_id="c1", tool_name="mock_high", arguments={})
    outcome = await ex.execute_call(call, task_id="t1")
    confirm_id = outcome.confirmation.confirmation_id

    # Approve and execute.
    store.approve(confirm_id)
    resumed = await ex.execute_confirmed(confirm_id, task_id="t1")
    assert resumed.result is not None
    assert resumed.result.success
    assert resumed.result.output == "high risk result"


async def test_executor_rejects_confirmation_not_approved() -> None:
    reg = ToolRegistry()
    await reg.register(MockHighRiskTool())
    store = ConfirmationStore()
    ex = ToolExecutor(registry=reg, policy=DefaultToolPolicy(), confirmation_store=store)
    call = ToolCallRecord(tool_call_id="c1", tool_name="mock_high", arguments={})
    outcome = await ex.execute_call(call, task_id="t1")
    confirm_id = outcome.confirmation.confirmation_id
    # Don't approve — try to execute.
    from app.tools.executor import ToolExecutionError

    with pytest.raises(ToolExecutionError):
        await ex.execute_confirmed(confirm_id, task_id="t1")


async def test_executor_confirmation_not_found() -> None:
    reg = ToolRegistry()
    ex = ToolExecutor(registry=reg, policy=DefaultToolPolicy())
    from app.tools.executor import ToolExecutionError

    with pytest.raises(ToolExecutionError):
        await ex.execute_confirmed("nonexistent", task_id="t1")


# --- ToolExecutor: repeated call protection ---


async def test_executor_records_call_history() -> None:
    reg = ToolRegistry()
    await _register_defaults(reg)
    ex = ToolExecutor(registry=reg, policy=DefaultToolPolicy())
    call = ToolCallRecord(tool_call_id="c1", tool_name="echo", arguments={"x": 1})
    ex.record_call("t1", call)
    assert ex.repeated_call_count("t1", call) == 1
    ex.record_call("t1", call)
    assert ex.repeated_call_count("t1", call) == 2


async def test_executor_repeated_call_count_differs_for_different_args() -> None:
    reg = ToolRegistry()
    await _register_defaults(reg)
    ex = ToolExecutor(registry=reg, policy=DefaultToolPolicy())
    call1 = ToolCallRecord(tool_call_id="c1", tool_name="echo", arguments={"x": 1})
    call2 = ToolCallRecord(tool_call_id="c2", tool_name="echo", arguments={"x": 2})
    ex.record_call("t1", call1)
    ex.record_call("t1", call2)
    assert ex.repeated_call_count("t1", call1) == 1
    assert ex.repeated_call_count("t1", call2) == 1


# --- ToolExecutor: events ---


async def test_executor_publishes_events() -> None:
    reg = ToolRegistry()
    await _register_defaults(reg)
    bus = InMemoryEventBus()
    ex = ToolExecutor(registry=reg, policy=DefaultToolPolicy(), event_bus=bus)
    events = []

    async def handler(ev) -> None:
        events.append(ev)

    bus.subscribe(None, handler)
    call = ToolCallRecord(tool_call_id="c1", tool_name="echo", arguments={})
    await ex.execute_call(call, task_id="t1")
    types = {e.event_type for e in events}
    assert EventType.TOOL_CALL_REQUESTED in types
    assert EventType.TOOL_CALL_STARTED in types
    assert EventType.TOOL_CALL_COMPLETED in types
    assert EventType.OBSERVATION_CREATED in types


async def test_executor_publishes_denial_event() -> None:
    reg = ToolRegistry()
    await _register_defaults(reg)
    bus = InMemoryEventBus()
    ex = ToolExecutor(registry=reg, policy=DefaultToolPolicy(deny={"echo"}), event_bus=bus)
    events = []

    async def handler(ev) -> None:
        events.append(ev)

    bus.subscribe(None, handler)
    call = ToolCallRecord(tool_call_id="c1", tool_name="echo", arguments={})
    await ex.execute_call(call, task_id="t1")
    types = {e.event_type for e in events}
    assert EventType.TOOL_POLICY_DENIED in types


async def test_executor_publishes_confirmation_event() -> None:
    reg = ToolRegistry()
    await reg.register(MockHighRiskTool())
    bus = InMemoryEventBus()
    store = ConfirmationStore()
    ex = ToolExecutor(
        registry=reg, policy=DefaultToolPolicy(), confirmation_store=store, event_bus=bus
    )
    events = []

    async def handler(ev) -> None:
        events.append(ev)

    bus.subscribe(None, handler)
    call = ToolCallRecord(tool_call_id="c1", tool_name="mock_high", arguments={})
    await ex.execute_call(call, task_id="t1")
    types = {e.event_type for e in events}
    assert EventType.TOOL_CONFIRMATION_REQUIRED in types


# --- ConfirmationStore ---


def test_confirmation_store_create_and_get() -> None:
    from app.tools.confirmation import ConfirmationRequest

    store = ConfirmationStore()
    req = ConfirmationRequest(tool_name="test", arguments={})
    store.create(req)
    assert store.get(req.confirmation_id) is not None


def test_confirmation_store_approve() -> None:
    from app.tools.confirmation import ConfirmationRequest

    store = ConfirmationStore()
    req = ConfirmationRequest(tool_name="test", arguments={})
    store.create(req)
    approved = store.approve(req.confirmation_id)
    assert approved is not None
    assert approved.is_approved


def test_confirmation_store_reject() -> None:
    from app.tools.confirmation import ConfirmationRequest

    store = ConfirmationStore()
    req = ConfirmationRequest(tool_name="test", arguments={})
    store.create(req)
    rejected = store.reject(req.confirmation_id)
    assert rejected is not None
    assert rejected.is_rejected


def test_confirmation_store_list_pending() -> None:
    from app.tools.confirmation import ConfirmationRequest

    store = ConfirmationStore()
    req = ConfirmationRequest(tool_name="test", arguments={})
    store.create(req)
    assert len(store.list_pending()) == 1
    store.approve(req.confirmation_id)
    assert len(store.list_pending()) == 0


def test_confirmation_store_approve_nonexistent_returns_none() -> None:
    store = ConfirmationStore()
    assert store.approve("nonexistent") is None


def test_confirmation_store_approve_already_decided_returns_none() -> None:
    from app.tools.confirmation import ConfirmationRequest

    store = ConfirmationStore()
    req = ConfirmationRequest(tool_name="test", arguments={})
    store.create(req)
    store.approve(req.confirmation_id)
    # Second approve should fail.
    assert store.approve(req.confirmation_id) is None
