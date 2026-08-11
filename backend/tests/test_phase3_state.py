"""Tests for LoopState Phase 3 extensions, LoopPolicy tool limits, and events."""

from __future__ import annotations

import pytest

from app.events import EventType
from app.runtime.loop.loop_policy import LoopPolicy
from app.runtime.loop.loop_state import LoopState
from app.runtime.loop.observation import Observation, ObservationType

# --- LoopState Phase 3 fields ---


def test_loop_state_has_tool_fields() -> None:
    state = LoopState(task_id="t1", goal="test")
    assert state.tool_call_count == 0
    assert state.tool_call_count_per_iteration == 0
    assert state.observations == []
    assert state.tool_results == []
    assert state.pending_tool_calls == []
    assert state.confirmation_required is False
    assert state.confirmation_request is None
    assert state.current_plan is None
    assert state.current_step is None


def test_loop_state_evolve_preserves_tool_fields() -> None:
    state = LoopState(task_id="t1", goal="test")
    state = state.evolve(tool_call_count=3, observations=[{"source": "tool:calc"}])
    assert state.tool_call_count == 3
    assert len(state.observations) == 1


# --- LoopPolicy tool limits ---


def test_loop_policy_default_tool_limits() -> None:
    policy = LoopPolicy()
    assert policy.max_tool_calls_per_iteration == 8
    assert policy.max_tool_calls_per_task == 32
    assert policy.max_repeated_identical_tool_calls == 3


def test_loop_policy_custom_tool_limits() -> None:
    policy = LoopPolicy(max_tool_calls_per_iteration=5, max_tool_calls_per_task=20)
    assert policy.max_tool_calls_per_iteration == 5
    assert policy.max_tool_calls_per_task == 20


def test_loop_policy_task_must_be_geq_iteration() -> None:
    from app.runtime.loop.loop_errors import LoopPolicyError

    with pytest.raises(LoopPolicyError):
        LoopPolicy(max_tool_calls_per_iteration=10, max_tool_calls_per_task=5)


# --- Observation model ---


def test_observation_from_tool_result_success() -> None:
    obs = Observation.from_tool_result(
        tool_name="calculator", tool_call_id="c1", result={"result": 42}, success=True
    )
    assert obs.type is ObservationType.TOOL_RESULT
    assert obs.success
    assert obs.tool_name == "calculator"
    assert obs.tool_call_id == "c1"


def test_observation_from_tool_result_failure() -> None:
    obs = Observation.from_tool_result(
        tool_name="calculator", tool_call_id="c1", result=None, success=False, error="bad expr"
    )
    assert obs.type is ObservationType.TOOL_FAILURE
    assert not obs.success
    assert obs.content == "bad expr"


def test_observation_from_denial() -> None:
    obs = Observation.from_denial(
        tool_name="mock_high", tool_call_id="c1", reason="denied by policy"
    )
    assert obs.type is ObservationType.TOOL_DENIED
    assert not obs.success
    assert "denied" in obs.content


def test_observation_from_confirmation_required() -> None:
    obs = Observation.from_confirmation_required(
        tool_name="mock_high",
        tool_call_id="c1",
        confirmation_id="conf_123",
        reason="confirm needed",
    )
    assert obs.type is ObservationType.TOOL_CONFIRMATION_REQUIRED
    assert obs.metadata["confirmation_id"] == "conf_123"


def test_observation_is_immutable() -> None:
    from pydantic import ValidationError

    obs = Observation(source="test", content="hello")
    with pytest.raises(ValidationError):
        obs.content = "changed"  # type: ignore[misc]


# --- Phase 3 event types ---


def test_phase3_event_types_exist() -> None:
    assert EventType.TOOL_CALL_REQUESTED.value == "tool.call.requested"
    assert EventType.TOOL_CALL_STARTED.value == "tool.call.started"
    assert EventType.TOOL_CALL_COMPLETED.value == "tool.call.completed"
    assert EventType.TOOL_CALL_FAILED.value == "tool.call.failed"
    assert EventType.TOOL_POLICY_DENIED.value == "tool.policy.denied"
    assert EventType.TOOL_CONFIRMATION_REQUIRED.value == "tool.confirmation.required"
    assert EventType.TOOL_CONFIRMATION_APPROVED.value == "tool.confirmation.approved"
    assert EventType.TOOL_CONFIRMATION_REJECTED.value == "tool.confirmation.rejected"
    assert EventType.OBSERVATION_CREATED.value == "observation.created"
    assert EventType.LLM_FALLBACK.value == "llm.fallback"
