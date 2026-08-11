"""Deterministic demonstration workflows for the Universal Loop Engine.

These prove the loop engine works end-to-end without any LLM. Each demo wires
concrete (deterministic) agents and verifiers through the real
:class:`LoopController` — the results are not faked.

* Workflow A — successful task (no iteration)
* Workflow B — one retry (first attempt wrong, second correct)
* Workflow C — maximum iterations (never succeeds)
* Workflow D — cancellation

They are exercised by the test suite and are also runnable directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import anyio

from app.agents.interface import (
    AgentCapability,
    AgentContext,
    AgentInfo,
    AgentInterface,
    AgentResult,
    AgentStatus,
)
from app.agents.registry import AgentRegistry
from app.events import EventBus, InMemoryEventBus
from app.runtime.loop.loop_context import LoopContext
from app.runtime.loop.loop_controller import LoopController
from app.runtime.loop.loop_policy import LoopPolicy
from app.runtime.loop.loop_result import LoopResult
from app.runtime.loop.loop_state import LoopState
from app.runtime.loop.stages import DefaultDiscoverStage, DefaultPlanStage
from app.runtime.loop.stages.execute import DefaultExecuteStage
from app.runtime.loop.stages.iterate import DefaultIterateStage
from app.runtime.loop.stages.verify import DefaultVerifyStage
from app.runtime.loop.verification import ExactMatchVerifier
from app.runtime.models import Task
from app.tools.registry import ToolRegistry

__all__ = [
    "ConstantAgent",
    "EchoAgent",
    "ScriptedAgent",
    "SlowConstantAgent",
    "LoopRunner",
    "build_demo_agents",
    "run_successful_workflow",
    "run_retry_workflow",
    "run_max_iterations_workflow",
    "run_cancellation_workflow",
]


class ConstantAgent(AgentInterface):
    """Always returns the same output."""

    def __init__(self, agent_id: str, value: Any) -> None:
        self._value = value
        self._info = AgentInfo(
            agent_id=agent_id,
            name=agent_id,
            description="Returns a constant value.",
            capabilities={AgentCapability.REASONING},
        )

    @property
    def info(self) -> AgentInfo:
        return self._info

    async def execute(self, context: AgentContext) -> AgentResult:
        return AgentResult(
            task_id=context.task_id,
            agent_id=self.agent_id,
            status=AgentStatus.COMPLETED,
            output=self._value,
        )


class EchoAgent(AgentInterface):
    """Echoes the task input back as its output."""

    def __init__(self, agent_id: str) -> None:
        self._info = AgentInfo(
            agent_id=agent_id,
            name=agent_id,
            description="Echoes the task input.",
            capabilities={AgentCapability.REASONING},
        )

    @property
    def info(self) -> AgentInfo:
        return self._info

    async def execute(self, context: AgentContext) -> AgentResult:
        return AgentResult(
            task_id=context.task_id,
            agent_id=self.agent_id,
            status=AgentStatus.COMPLETED,
            output=context.input,
        )


class ScriptedAgent(AgentInterface):
    """Returns a scripted sequence of outputs, one per invocation per task.

    Used by the retry demo: the first call returns an incorrect value and the
    second returns the correct one. This makes iteration real — the verifier
    detects the wrong first result and the loop iterates.

    The script position is tracked per ``task_id`` so a single shared agent
    instance (as registered in the runtime) behaves correctly for every task,
    not just the first one.
    """

    def __init__(self, agent_id: str, outputs: list[Any]) -> None:
        self._outputs = list(outputs)
        self._positions: dict[str, int] = {}
        self._info = AgentInfo(
            agent_id=agent_id,
            name=agent_id,
            description="Returns a scripted sequence of outputs.",
            capabilities={AgentCapability.REASONING},
        )

    @property
    def info(self) -> AgentInfo:
        return self._info

    async def execute(self, context: AgentContext) -> AgentResult:
        index = self._positions.get(context.task_id, 0)
        if index >= len(self._outputs):
            value = self._outputs[-1]
        else:
            value = self._outputs[index]
            self._positions[context.task_id] = index + 1
        return AgentResult(
            task_id=context.task_id,
            agent_id=self.agent_id,
            status=AgentStatus.COMPLETED,
            output=value,
        )


class SlowConstantAgent(ConstantAgent):
    """A ConstantAgent that sleeps before returning.

    Used to demonstrate cooperative cancellation via the public REST API: a
    background loop with a high iteration budget stays running long enough for
    an external ``POST /tasks/{id}/cancel`` request to be honoured at the next
    stage boundary.
    """

    def __init__(self, agent_id: str, value: Any, *, delay_seconds: float = 0.3) -> None:
        super().__init__(agent_id, value)
        self._delay = delay_seconds
        self._info = AgentInfo(
            agent_id=agent_id,
            name=agent_id,
            description="Returns a constant value after a short delay.",
            capabilities={AgentCapability.REASONING},
        )

    async def execute(self, context: AgentContext) -> AgentResult:
        await anyio.sleep(self._delay)
        return await super().execute(context)


def build_demo_agents() -> list[AgentInterface]:
    """Construct the deterministic demo agents used by the running runtime.

    These are registered at application startup (non-production) so the
    Universal Loop Engine can be exercised end-to-end through the REST API
    without an LLM:

    * ``demo.echo``        — echoes its input (success workflow).
    * ``demo.math``        — returns 99 then 100 (retry workflow).
    * ``demo.failing``     — always returns 99 (max-iteration workflow).
    * ``demo.cancel``      — slow, never-verifying (cancellation workflow).
    """
    return [
        EchoAgent("demo.echo"),
        ScriptedAgent("demo.math", outputs=[99, 100]),
        ConstantAgent("demo.failing", 99),
        SlowConstantAgent("demo.cancel", 99, delay_seconds=0.3),
    ]


@dataclass
class LoopRunner:
    """Wires registries, event bus, and a :class:`LoopController`."""

    agent_registry: AgentRegistry = field(default_factory=AgentRegistry)
    tool_registry: ToolRegistry = field(default_factory=ToolRegistry)
    event_bus: EventBus = field(default_factory=InMemoryEventBus)

    async def register(self, agent: AgentInterface) -> AgentInterface:
        return await self.agent_registry.register(agent)

    def make_controller(
        self,
        *,
        verifier=None,
        max_iterations: int = 5,
        expected_output: Any = None,
    ) -> LoopController:
        policy = LoopPolicy(max_iterations=max_iterations)
        return LoopController(
            discover_stage=DefaultDiscoverStage(),
            plan_stage=DefaultPlanStage(),
            execute_stage=DefaultExecuteStage(),
            verify_stage=DefaultVerifyStage(verifier=verifier),
            iterate_stage=DefaultIterateStage(),
        ), policy

    def make_context(
        self,
        task: Task,
        state: LoopState,
        policy: LoopPolicy,
        expected_output: Any = None,
    ) -> LoopContext:
        return LoopContext(
            state=state,
            task=task,
            agent_registry=self.agent_registry,
            tool_registry=self.tool_registry,
            event_bus=self.event_bus,
            policy=policy,
            stage_config={"expected_output": expected_output},
        )


def _make_task(agent_id: str, goal: str) -> Task:
    return Task(input=goal, agent_id=agent_id, metadata={})


async def run_successful_workflow() -> LoopResult:
    """Workflow A — a task that succeeds on the first attempt."""
    runner = LoopRunner()
    agent = ConstantAgent("hello-agent", "hello")
    await runner.register(agent)

    verifier = ExactMatchVerifier(check_name="returns_hello")
    controller, policy = runner.make_controller(
        verifier=verifier, max_iterations=5, expected_output="hello"
    )
    task = _make_task("hello-agent", "Return the word hello.")
    state = LoopState(
        task_id=task.task_id,
        goal="Return the word hello.",
        success_criteria=["expected:hello"],
        max_iterations=5,
    )
    context = runner.make_context(task, state, policy, expected_output="hello")
    return await controller.run(context)


async def run_retry_workflow() -> LoopResult:
    """Workflow B — first attempt returns 99, second returns 100."""
    runner = LoopRunner()
    # Scripted: wrong on first call, correct on the second.
    agent = ScriptedAgent("counter-agent", outputs=[99, 100])
    await runner.register(agent)

    verifier = ExactMatchVerifier(check_name="equals_100")
    controller, policy = runner.make_controller(
        verifier=verifier, max_iterations=5, expected_output=100
    )
    task = _make_task("counter-agent", "Return 100.")
    state = LoopState(
        task_id=task.task_id,
        goal="Return 100.",
        success_criteria=["expected:100"],
        max_iterations=5,
    )
    context = runner.make_context(task, state, policy, expected_output=100)
    return await controller.run(context)


async def run_max_iterations_workflow(max_iterations: int = 3) -> LoopResult:
    """Workflow C — never succeeds; always returns 99."""
    runner = LoopRunner()
    agent = ConstantAgent("failing-counter", 99)
    await runner.register(agent)

    verifier = ExactMatchVerifier(check_name="equals_100")
    controller, policy = runner.make_controller(
        verifier=verifier, max_iterations=max_iterations, expected_output=100
    )
    task = _make_task("failing-counter", "Return 100.")
    state = LoopState(
        task_id=task.task_id,
        goal="Return 100.",
        success_criteria=["expected:100"],
        max_iterations=max_iterations,
    )
    context = runner.make_context(task, state, policy, expected_output=100)
    return await controller.run(context)


async def run_cancellation_workflow() -> LoopResult:
    """Workflow D — a task cancelled before execution completes.

    Cancellation is requested via a custom execute stage that cancels the loop
    after the first execution, proving no further steps begin.
    """

    class _CancelAfterFirstExecute(DefaultExecuteStage):
        async def execute(self, context: LoopContext) -> Any:
            result = await super().execute(context)
            context.state = context.state.request_cancel()
            return result

    runner = LoopRunner()
    agent = ConstantAgent("cancel-agent", "hello")
    await runner.register(agent)

    verifier = ExactMatchVerifier(check_name="returns_hello")
    controller = LoopController(
        discover_stage=DefaultDiscoverStage(),
        plan_stage=DefaultPlanStage(),
        execute_stage=_CancelAfterFirstExecute(),
        verify_stage=DefaultVerifyStage(verifier=verifier),
        iterate_stage=DefaultIterateStage(),
    )
    policy = LoopPolicy(max_iterations=5)
    task = _make_task("cancel-agent", "Return hello (will be cancelled).")
    state = LoopState(
        task_id=task.task_id,
        goal="Return hello (will be cancelled).",
        success_criteria=["expected:hello"],
        max_iterations=5,
    )
    context = runner.make_context(task, state, policy, expected_output="hello")
    return await controller.run(context)
