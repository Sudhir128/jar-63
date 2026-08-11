"""Tests for loop state, stop conditions, and policy."""

from __future__ import annotations

import pytest

from app.runtime.loop.conditions import (
    CancellationCondition,
    MaxIterationsCondition,
    StopDecision,
    SuccessCondition,
    default_stop_conditions,
)
from app.runtime.loop.loop_errors import LoopPolicyError
from app.runtime.loop.loop_policy import LoopPolicy
from app.runtime.loop.loop_result import LoopFinalStatus
from app.runtime.loop.loop_state import (
    ActionType,
    IterationRecord,
    LoopState,
    LoopStatus,
    NextAction,
    StageStatus,
)
from app.runtime.loop.verification import VerificationResult, VerificationStatus


def _state(**kwargs) -> LoopState:
    base = {"task_id": "t1", "goal": "g", "max_iterations": 5}
    base.update(kwargs)
    return LoopState(**base)


def test_loop_state_defaults() -> None:
    s = _state()
    assert s.status is LoopStatus.CREATED
    assert s.iteration_count == 0
    assert s.max_iterations == 5
    assert s.loop_id.startswith("loop_")
    assert s.is_terminal is False


def test_loop_state_evolve_is_functional() -> None:
    s = _state()
    s2 = s.evolve(iteration_count=3, goal="new goal")
    assert s2.iteration_count == 3
    assert s2.goal == "new goal"
    # Original is unchanged.
    assert s.iteration_count == 0
    assert s.goal == "g"
    assert s2.updated_at >= s.updated_at


def test_loop_state_begin_iteration_increments() -> None:
    s = _state()
    s2 = s.begin_iteration()
    assert s2.iteration_count == 1
    assert s2.status is LoopStatus.ITERATING


def test_loop_state_request_cancel() -> None:
    s = _state().request_cancel()
    assert s.cancel_requested is True
    assert s.status is LoopStatus.CANCELLED


def test_loop_state_iteration_history_append_only() -> None:
    s = _state()
    rec = IterationRecord(iteration_number=1, stage="execute")
    s2 = s.add_iteration_record(rec)
    assert len(s.iteration_history) == 0
    assert len(s2.iteration_history) == 1


def test_terminal_statuses() -> None:
    for status in (
        LoopStatus.COMPLETED,
        LoopStatus.FAILED,
        LoopStatus.CANCELLED,
        LoopStatus.MAX_ITERATIONS_REACHED,
    ):
        assert _state(status=status).is_terminal is True


def test_policy_defaults() -> None:
    p = LoopPolicy()
    assert p.max_iterations == 5
    assert p.require_verification is True


def test_policy_rejects_invalid_task_time() -> None:
    with pytest.raises(LoopPolicyError):
        LoopPolicy(max_execution_time_seconds=300, max_task_time_seconds=100)


def test_policy_rejects_retries_without_allow() -> None:
    with pytest.raises(LoopPolicyError):
        LoopPolicy(allow_retry=False, max_retries_per_step=1)


def test_stop_decision_continue() -> None:
    d = StopDecision.continue_loop()
    assert d.should_stop is False


def test_max_iterations_condition() -> None:
    from app.runtime.loop.loop_context import LoopContext

    s = _state(iteration_count=5, max_iterations=5)
    p = LoopPolicy()
    ctx = LoopContext(
        state=s, task=None, agent_registry=None, tool_registry=None, event_bus=None, policy=p
    )  # type: ignore[arg-type]
    d = MaxIterationsCondition().evaluate(ctx)
    assert d.should_stop is True
    assert d.status is LoopFinalStatus.MAX_ITERATIONS_REACHED


def test_max_iterations_condition_not_reached() -> None:
    from app.runtime.loop.loop_context import LoopContext

    s = _state(iteration_count=2, max_iterations=5)
    ctx = LoopContext(
        state=s,
        task=None,
        agent_registry=None,
        tool_registry=None,
        event_bus=None,
        policy=LoopPolicy(),
    )  # type: ignore[arg-type]
    assert MaxIterationsCondition().evaluate(ctx).should_stop is False


def test_success_condition_requires_evidence() -> None:
    from app.runtime.loop.loop_context import LoopContext

    # Passed but no evidence -> not success (no objective evidence).
    s = _state(last_verification=VerificationResult(status=VerificationStatus.PASSED, evidence=[]))
    ctx = LoopContext(
        state=s,
        task=None,
        agent_registry=None,
        tool_registry=None,
        event_bus=None,
        policy=LoopPolicy(),
    )  # type: ignore[arg-type]
    assert SuccessCondition().evaluate(ctx).should_stop is False

    from app.runtime.loop.verification import VerificationEvidence

    s2 = _state(
        last_verification=VerificationResult(
            status=VerificationStatus.PASSED,
            evidence=[VerificationEvidence(check="c", passed=True)],
        )
    )
    ctx2 = LoopContext(
        state=s2,
        task=None,
        agent_registry=None,
        tool_registry=None,
        event_bus=None,
        policy=LoopPolicy(),
    )  # type: ignore[arg-type]
    d = SuccessCondition().evaluate(ctx2)
    assert d.should_stop is True
    assert d.status is LoopFinalStatus.SUCCESS


def test_cancellation_condition() -> None:
    from app.runtime.loop.loop_context import LoopContext

    s = _state(cancel_requested=True)
    ctx = LoopContext(
        state=s,
        task=None,
        agent_registry=None,
        tool_registry=None,
        event_bus=None,
        policy=LoopPolicy(),
    )  # type: ignore[arg-type]
    d = CancellationCondition().evaluate(ctx)
    assert d.should_stop is True
    assert d.status is LoopFinalStatus.CANCELLED


def test_default_stop_conditions_order() -> None:
    conds = default_stop_conditions()
    # Cancellation and success must come before max iterations.
    types = [type(c).__name__ for c in conds]
    assert types.index("CancellationCondition") < types.index("MaxIterationsCondition")
    assert types.index("SuccessCondition") < types.index("MaxIterationsCondition")


def test_action_type_values() -> None:
    assert ActionType.EXECUTE_AGENT.value == "execute_agent"
    assert NextAction(action_type=ActionType.COMPLETE).action_type is ActionType.COMPLETE


def test_stage_status_values() -> None:
    assert StageStatus.DISCOVER.value == "discover"
    assert StageStatus.DONE.value == "done"
