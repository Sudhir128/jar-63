"""Tests for the LangGraph adapter boundary.

Verifies the loop produces identical outcomes whether driven by the
:class:`LoopController` directly or through the LangGraph state machine.
"""

from __future__ import annotations

import pytest

from app.runtime.loop.demos import ConstantAgent, LoopRunner, ScriptedAgent, _make_task
from app.runtime.loop.graph import LoopGraphAdapter
from app.runtime.loop.loop_controller import LoopController
from app.runtime.loop.loop_policy import LoopPolicy
from app.runtime.loop.loop_result import LoopFinalStatus
from app.runtime.loop.loop_state import LoopState
from app.runtime.loop.stages import (
    DefaultDiscoverStage,
    DefaultExecuteStage,
    DefaultIterateStage,
    DefaultPlanStage,
    DefaultVerifyStage,
)
from app.runtime.loop.verification import ExactMatchVerifier

pytestmark = pytest.mark.asyncio


def _controller(verifier=None) -> LoopController:
    return LoopController(
        discover_stage=DefaultDiscoverStage(),
        plan_stage=DefaultPlanStage(),
        execute_stage=DefaultExecuteStage(),
        verify_stage=DefaultVerifyStage(verifier=verifier),
        iterate_stage=DefaultIterateStage(),
    )


async def test_langgraph_success_workflow() -> None:
    runner = LoopRunner()
    agent = ConstantAgent("hello-agent", "hello")
    await runner.register(agent)
    controller = _controller(ExactMatchVerifier())
    task = _make_task("hello-agent", "Return hello.")
    state = LoopState(
        task_id=task.task_id,
        goal="Return hello.",
        success_criteria=["expected:hello"],
        max_iterations=5,
    )
    ctx = runner.make_context(task, state, LoopPolicy(), expected_output="hello")

    adapter = LoopGraphAdapter(controller)
    result = await adapter.run(ctx)
    assert result.final_status is LoopFinalStatus.SUCCESS
    assert result.success is True
    assert result.iterations_used >= 1
    assert result.final_response == "hello"


async def test_langgraph_retry_workflow() -> None:
    runner = LoopRunner()
    agent = ScriptedAgent("counter-agent", outputs=[99, 100])
    await runner.register(agent)
    controller = _controller(ExactMatchVerifier())
    task = _make_task("counter-agent", "Return 100.")
    state = LoopState(
        task_id=task.task_id,
        goal="Return 100.",
        success_criteria=["expected:100"],
        max_iterations=5,
    )
    ctx = runner.make_context(task, state, LoopPolicy(), expected_output=100)

    adapter = LoopGraphAdapter(controller)
    result = await adapter.run(ctx)
    assert result.final_status is LoopFinalStatus.SUCCESS
    assert result.final_response == 100


async def test_langgraph_max_iterations_workflow() -> None:
    runner = LoopRunner()
    agent = ConstantAgent("failing-counter", 99)
    await runner.register(agent)
    controller = _controller(ExactMatchVerifier())
    task = _make_task("failing-counter", "Return 100.")
    state = LoopState(
        task_id=task.task_id,
        goal="Return 100.",
        success_criteria=["expected:100"],
        max_iterations=3,
    )
    ctx = runner.make_context(task, state, LoopPolicy(max_iterations=3), expected_output=100)

    adapter = LoopGraphAdapter(controller)
    result = await adapter.run(ctx)
    assert result.final_status is LoopFinalStatus.MAX_ITERATIONS_REACHED
    assert result.success is False


async def test_langgraph_adapter_builds_compiled_graph() -> None:
    controller = _controller()
    adapter = LoopGraphAdapter(controller)
    graph = adapter.graph
    # The compiled graph exposes ainvoke.
    assert hasattr(graph, "ainvoke")
