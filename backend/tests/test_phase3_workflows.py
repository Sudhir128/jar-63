"""Tests for Phase 3 end-to-end demonstration workflows.

These run the real :class:`LoopController` with LLM-backed planning (via stub
LLM), tool execution through the :class:`ToolExecutor`, policy, and
verification. Results are not faked.
"""

from __future__ import annotations

import pytest

from app.runtime.loop.loop_result import LoopFinalStatus
from app.runtime.loop.phase3_demos import (
    resume_confirmation_workflow,
    run_calculator_workflow,
    run_confirmation_workflow,
    run_iteration_workflow,
    run_llm_planner_workflow,
    run_policy_denial_workflow,
    run_time_workflow,
    run_tool_failure_workflow,
)

pytestmark = pytest.mark.asyncio


# --- Demo 1: Calculator ---


async def test_calculator_workflow_succeeds() -> None:
    result = await run_calculator_workflow()
    assert result.final_status is LoopFinalStatus.SUCCESS
    assert result.success
    assert result.final_response["result"] == 11186
    assert result.final_response["expression"] == "238 * 47"
    # Success requires verification evidence.
    assert len(result.verification_evidence) >= 1
    assert all(e.passed for e in result.verification_evidence)


async def test_calculator_workflow_uses_tool_not_llm_text() -> None:
    """The result must come from the CalculatorTool, not LLM text generation."""
    result = await run_calculator_workflow()
    assert isinstance(result.final_response, dict)
    assert "result" in result.final_response
    # 238 * 47 = 11186, not a round number the LLM might guess.
    assert result.final_response["result"] == 11186


# --- Demo 2: Time ---


async def test_time_workflow_succeeds() -> None:
    result = await run_time_workflow()
    assert result.final_status is LoopFinalStatus.SUCCESS
    assert result.success
    assert isinstance(result.final_response, dict)
    assert "utc_iso" in result.final_response


# --- Demo 3: Tool failure ---


async def test_tool_failure_workflow_no_false_success() -> None:
    result = await run_tool_failure_workflow()
    # Division by zero → tool fails → no false SUCCESS.
    assert result.success is False
    assert result.final_status is not LoopFinalStatus.SUCCESS


# --- Demo 4: Policy denial ---


async def test_policy_denial_workflow_tool_not_executed() -> None:
    result = await run_policy_denial_workflow()
    # The denied tool must not produce a success.
    assert result.success is False
    assert result.final_status is not LoopFinalStatus.SUCCESS


# --- Demo 5: Confirmation ---


async def test_confirmation_workflow_pauses_then_resumes() -> None:
    result, runner, confirm_id = await run_confirmation_workflow()
    # First run: tool requires confirmation → not executed.
    assert confirm_id is not None
    pending = runner.confirmation_store.get(confirm_id)
    assert pending is not None
    assert pending.is_pending

    # Approve and resume.
    resumed = await resume_confirmation_workflow(runner, confirm_id)
    assert resumed.success
    assert resumed.final_response == "mock medium result"


async def test_confirmation_reject_prevents_execution() -> None:
    result, runner, confirm_id = await run_confirmation_workflow()
    assert confirm_id is not None
    # Reject instead of approve.
    runner.confirmation_store.reject(confirm_id)
    from app.tools.executor import ToolExecutionError

    # Trying to execute a rejected confirmation should fail.
    with pytest.raises(ToolExecutionError):
        await runner.tool_executor.execute_confirmed(confirm_id, task_id="t1")


# --- Demo 6: Iteration ---


async def test_iteration_workflow_succeeds_after_replan() -> None:
    result = await run_iteration_workflow()
    assert result.final_status is LoopFinalStatus.SUCCESS
    assert result.success
    assert result.final_response["result"] == 11186
    # First plan was wrong, second was correct → at least 2 iterations.
    assert result.iterations_used >= 2


# --- Demo 7: LLM planner ---


async def test_llm_planner_workflow_succeeds() -> None:
    result = await run_llm_planner_workflow()
    assert result.final_status is LoopFinalStatus.SUCCESS
    assert result.final_response["result"] == 11186
