"""Phase 4 tests: Tool policy & executor confirmation behavior.

Tests cover:
* DefaultToolPolicy: LOW risk auto-approved, MEDIUM requires confirmation
  (when enabled), HIGH always requires confirmation.
* ToolExecutor: execute_call returns confirmation outcome for MEDIUM.
* ToolExecutor: execute_confirmed runs the tool after approval.
* ToolExecutor: rejected/expired confirmation is not executed.
* Confirmation expiration in executor flow.
"""

from __future__ import annotations

import pytest

from app.events import InMemoryEventBus
from app.tools.confirmation import ConfirmationStore
from app.tools.executor import ToolCallRecord, ToolExecutor
from app.tools.impl import CalculatorTool
from app.tools.interface import RiskLevel, ToolInfo
from app.tools.policy import AllowAllToolPolicy, DefaultToolPolicy
from app.tools.registry import ToolRegistry


def _make_executor(*, policy=None) -> tuple[ToolExecutor, ToolRegistry, ConfirmationStore]:
    store = ConfirmationStore()
    registry = ToolRegistry()
    executor = ToolExecutor(
        registry=registry,
        policy=policy or AllowAllToolPolicy(),
        confirmation_store=store,
        event_bus=InMemoryEventBus(),
    )
    return executor, registry, store


async def _register_calc(registry: ToolRegistry, *, risk: RiskLevel = RiskLevel.LOW) -> None:
    await registry.register(CalculatorTool())
    calc = registry.get("calculator")
    calc._info = calc.info.model_copy(update={"risk_level": risk})


def _calc_info(*, risk: RiskLevel = RiskLevel.LOW) -> ToolInfo:
    return CalculatorTool().info.model_copy(update={"risk_level": risk})


def _call(
    *, name: str = "calculator", args: dict | None = None, cid: str = "call_1"
) -> ToolCallRecord:
    return ToolCallRecord(
        tool_call_id=cid, tool_name=name, arguments=args or {"expression": "2 + 2"}
    )


# --- Policy decisions ---


def test_allow_all_policy_auto_approves_high_risk() -> None:
    policy = AllowAllToolPolicy()
    decision = policy.evaluate(_calc_info(risk=RiskLevel.HIGH))
    assert decision.allowed is True


def test_default_policy_auto_approves_low_risk() -> None:
    policy = DefaultToolPolicy()
    decision = policy.evaluate(_calc_info(risk=RiskLevel.LOW))
    assert decision.allowed is True


def test_default_policy_requires_confirmation_for_medium_when_enabled() -> None:
    policy = DefaultToolPolicy(require_confirmation_for_medium=True)
    decision = policy.evaluate(_calc_info(risk=RiskLevel.MEDIUM))
    assert decision.allowed is False
    assert decision.requires_confirmation is True


def test_default_policy_auto_approves_medium_when_disabled() -> None:
    policy = DefaultToolPolicy(require_confirmation_for_medium=False)
    decision = policy.evaluate(_calc_info(risk=RiskLevel.MEDIUM))
    assert decision.allowed is True


def test_default_policy_always_requires_confirmation_for_high() -> None:
    policy = DefaultToolPolicy(require_confirmation_for_medium=False)
    decision = policy.evaluate(_calc_info(risk=RiskLevel.HIGH))
    assert decision.allowed is False
    assert decision.requires_confirmation is True


def test_default_policy_denies_explicit_deny_list() -> None:
    policy = DefaultToolPolicy(deny={"calculator"})
    decision = policy.evaluate(_calc_info(risk=RiskLevel.LOW))
    assert decision.denied is True


# --- Executor confirmation flow ---


async def test_executor_low_risk_executes_directly() -> None:
    executor, registry, store = _make_executor(policy=DefaultToolPolicy())
    await _register_calc(registry, risk=RiskLevel.LOW)
    outcome = await executor.execute_call(
        _call(args={"expression": "2 + 2"}),
        task_id="task_1",
        session_id="sess_1",
    )
    assert outcome.result is not None
    assert outcome.result.success is True
    assert outcome.result.output["result"] == 4
    assert outcome.skipped is False
    assert outcome.confirmation is None
    assert len(store) == 0


async def test_executor_medium_risk_returns_confirmation() -> None:
    executor, registry, store = _make_executor(
        policy=DefaultToolPolicy(require_confirmation_for_medium=True)
    )
    await _register_calc(registry, risk=RiskLevel.MEDIUM)
    outcome = await executor.execute_call(
        _call(args={"expression": "2 + 2"}),
        task_id="task_1",
        session_id="sess_1",
    )
    assert outcome.result is None
    assert outcome.skipped is True
    assert outcome.confirmation is not None
    assert outcome.confirmation.tool_name == "calculator"
    assert outcome.confirmation.status.value == "pending"
    assert len(store) == 1


async def test_executor_confirmed_executes_after_approval() -> None:
    executor, registry, store = _make_executor(
        policy=DefaultToolPolicy(require_confirmation_for_medium=True)
    )
    await _register_calc(registry, risk=RiskLevel.MEDIUM)
    outcome = await executor.execute_call(
        _call(args={"expression": "3 * 7"}),
        task_id="task_1",
        session_id="sess_1",
    )
    confirm_id = outcome.confirmation.confirmation_id
    store.approve(confirm_id)
    confirmed = await executor.execute_confirmed(confirm_id, task_id="task_1", session_id="sess_1")
    assert confirmed.result is not None
    assert confirmed.result.success is True
    assert confirmed.result.output["result"] == 21
    assert confirmed.skipped is False


async def test_executor_confirmed_rejected_raises() -> None:
    """Executing a rejected confirmation raises (it's not approved)."""
    from app.tools.executor import ToolExecutionError

    executor, registry, store = _make_executor(
        policy=DefaultToolPolicy(require_confirmation_for_medium=True)
    )
    await _register_calc(registry, risk=RiskLevel.MEDIUM)
    outcome = await executor.execute_call(
        _call(args={"expression": "3 * 7"}),
        task_id="task_1",
        session_id="sess_1",
    )
    confirm_id = outcome.confirmation.confirmation_id
    store.reject(confirm_id)
    with pytest.raises(ToolExecutionError):
        await executor.execute_confirmed(confirm_id)


async def test_executor_confirmed_expired_raises() -> None:
    from app.tools.executor import ToolExecutionError

    executor, registry, store = _make_executor(
        policy=DefaultToolPolicy(require_confirmation_for_medium=True)
    )
    await _register_calc(registry, risk=RiskLevel.MEDIUM)
    outcome = await executor.execute_call(
        _call(args={"expression": "3 * 7"}),
        task_id="task_1",
        session_id="sess_1",
    )
    confirm_id = outcome.confirmation.confirmation_id
    store.expire(confirm_id)
    with pytest.raises(ToolExecutionError):
        await executor.execute_confirmed(confirm_id)


async def test_executor_high_risk_returns_confirmation() -> None:
    executor, registry, store = _make_executor(
        policy=DefaultToolPolicy(require_confirmation_for_medium=False)
    )
    await _register_calc(registry, risk=RiskLevel.HIGH)
    outcome = await executor.execute_call(
        _call(args={"expression": "2 + 2"}),
        task_id="task_1",
        session_id="sess_1",
    )
    assert outcome.result is None
    assert outcome.confirmation is not None
    assert outcome.confirmation.risk_level is RiskLevel.HIGH


async def test_executor_nonexistent_tool_skipped() -> None:
    executor, _, _ = _make_executor(policy=DefaultToolPolicy())
    outcome = await executor.execute_call(
        _call(name="nonexistent"),
        task_id="task_1",
        session_id="sess_1",
    )
    assert outcome.result is None
    assert outcome.skipped is True
    assert "not registered" in (outcome.skipped_reason or "").lower()


async def test_executor_invalid_arguments_skipped() -> None:
    executor, registry, _ = _make_executor(policy=DefaultToolPolicy())
    await _register_calc(registry, risk=RiskLevel.LOW)
    outcome = await executor.execute_call(
        _call(args={"expression": 12345}),  # wrong type: int not str
        task_id="task_1",
        session_id="sess_1",
    )
    assert outcome.result is None
    assert outcome.skipped is True
