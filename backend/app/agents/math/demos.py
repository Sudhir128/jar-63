"""Phase 4 demonstration workflows: Math Agent + confirmation lifecycle.

These prove the Phase 4 features work end-to-end through the real
:class:`LoopController` and :class:`LoopService`, with real tools and real
verification. The results are not faked.

Demos:
* Demo 1 — ``238 * 47`` → 11186 (basic arithmetic)
* Demo 2 — ``25% of 800`` → 200 (percentage)
* Demo 3 — ``(25 + 15) * 3`` → 120 (parentheses)
* Demo 4 — ``10 / 0`` → structured failure (division by zero)
* Demo 5 — tool failure (simulated CalculatorTool failure)
* Demo 6 — confirmation approval → resume
* Demo 7 — confirmation rejection → tool not executed
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.agents.math import MATH_AGENT_ID, MathAgent
from app.agents.math.verifier import MathVerifier
from app.agents.registry import AgentRegistry
from app.events import EventBus, InMemoryEventBus
from app.runtime.loop.loop_context import LoopContext
from app.runtime.loop.loop_controller import LoopController
from app.runtime.loop.loop_policy import LoopPolicy
from app.runtime.loop.loop_result import LoopFinalStatus, LoopResult
from app.runtime.loop.loop_state import LoopState
from app.runtime.loop.stages import (
    DefaultDiscoverStage,
    DefaultIterateStage,
    DefaultPlanStage,
    DefaultVerifyStage,
)
from app.runtime.loop.stages.execute import DefaultExecuteStage
from app.runtime.models import Task
from app.tools.confirmation import ConfirmationStore
from app.tools.executor import ToolExecutor
from app.tools.impl import CalculatorTool
from app.tools.policy import AllowAllToolPolicy, DefaultToolPolicy, ToolPolicy
from app.tools.registry import ToolRegistry

__all__ = [
    "MathLoopRunner",
    "run_math_demo",
    "run_division_by_zero_demo",
    "run_tool_failure_demo",
    "run_confirmation_approval_demo",
    "run_confirmation_rejection_demo",
]


@dataclass
class MathLoopRunner:
    """Wires registries, tool executor, event bus, and a controller for math demos."""

    agent_registry: AgentRegistry = field(default_factory=AgentRegistry)
    tool_registry: ToolRegistry = field(default_factory=ToolRegistry)
    event_bus: EventBus = field(default_factory=InMemoryEventBus)
    confirmation_store: ConfirmationStore = field(default_factory=ConfirmationStore)
    tool_policy: ToolPolicy = field(default_factory=AllowAllToolPolicy)
    tool_executor: ToolExecutor | None = None
    controller: LoopController | None = None
    context: LoopContext | None = None
    _task: Task | None = None

    async def setup(self, *, policy: ToolPolicy | None = None) -> None:
        if policy is not None:
            self.tool_policy = policy
        self.tool_executor = ToolExecutor(
            registry=self.tool_registry,
            policy=self.tool_policy,
            confirmation_store=self.confirmation_store,
            event_bus=self.event_bus,
        )
        await self.tool_registry.register(CalculatorTool())
        await self.agent_registry.register(MathAgent(tool_executor=self.tool_executor))
        self.controller = self._make_controller()

    def _make_controller(self) -> LoopController:
        return LoopController(
            discover_stage=DefaultDiscoverStage(),
            plan_stage=DefaultPlanStage(),
            execute_stage=DefaultExecuteStage(tool_executor=self.tool_executor),
            verify_stage=DefaultVerifyStage(verifier=MathVerifier()),
            iterate_stage=DefaultIterateStage(),
        )

    async def run(
        self,
        *,
        goal: str,
        max_iterations: int = 5,
    ) -> LoopResult:
        assert self.controller is not None
        task = Task(input=goal, agent_id=MATH_AGENT_ID, metadata={})
        self._task = task
        policy = LoopPolicy(max_iterations=max_iterations, per_execution_timeout_seconds=10)
        state = LoopState(
            task_id=task.task_id,
            goal=goal,
            success_criteria=["math_verified"],
            max_iterations=max_iterations,
        )
        self.context = LoopContext(
            state=state,
            task=task,
            agent_registry=self.agent_registry,
            tool_registry=self.tool_registry,
            event_bus=self.event_bus,
            policy=policy,
            stage_config={},
        )
        return await self.controller.run(self.context)


async def run_math_demo(expression: str, *, expected: Any = None) -> LoopResult:
    """Run a math demo with the given user expression."""
    runner = MathLoopRunner()
    await runner.setup()
    result = await runner.run(goal=f"What is {expression}?")
    return result


async def run_division_by_zero_demo() -> LoopResult:
    """Demo 4 — division by zero produces a structured failure."""
    runner = MathLoopRunner()
    await runner.setup()
    return await runner.run(goal="Calculate 10 / 0")


async def run_tool_failure_demo() -> LoopResult:
    """Demo 5 — simulated CalculatorTool failure handled safely.

    Registers a custom tool that always fails, then runs the math agent.
    The agent must handle the failure gracefully — no fabricated answer.
    """
    from app.tools.interface import (
        RiskLevel,
        ToolCategory,
        ToolContext,
        ToolInfo,
        ToolInterface,
        ToolResult,
    )

    class FailingCalculatorTool(ToolInterface):
        """A calculator-shaped tool that always fails."""

        def __init__(self) -> None:
            self._info = ToolInfo(
                tool_id="calculator",
                name="calculator",
                description="Always-failing calculator for testing.",
                category=ToolCategory.CALCULATOR,
                risk_level=RiskLevel.LOW,
                input_schema={
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            )

        @property
        def info(self) -> ToolInfo:
            return self._info

        async def execute(self, context: ToolContext) -> ToolResult:
            return ToolResult(
                invocation_id=context.invocation_id,
                tool_call_id=context.tool_call_id,
                name="calculator",
                success=False,
                error="Simulated calculator failure.",
            )

    runner = MathLoopRunner()
    # Register the failing tool BEFORE setup so setup's CalculatorTool fails.
    await runner.tool_registry.register(FailingCalculatorTool())
    runner.tool_executor = ToolExecutor(
        registry=runner.tool_registry,
        policy=AllowAllToolPolicy(),
        confirmation_store=runner.confirmation_store,
        event_bus=runner.event_bus,
    )
    await runner.agent_registry.register(MathAgent(tool_executor=runner.tool_executor))
    runner.controller = runner._make_controller()
    return await runner.run(goal="What is 2 + 2?")


async def run_confirmation_approval_demo() -> tuple[LoopResult, LoopResult]:
    """Demo 6 — confirmation required, then approved, then resumed.

    Returns (paused_result, resumed_result). The loop pauses on the first
    attempt because the calculator is configured as MEDIUM-risk (requiring
    confirmation). After approval, the same loop resumes.
    """
    runner = MathLoopRunner()
    # Use a policy that requires confirmation for MEDIUM tools.
    await runner.setup(
        policy=DefaultToolPolicy(require_confirmation_for_medium=True)
    )
    # Override: make calculator MEDIUM risk so confirmation is triggered.
    from app.tools.interface import RiskLevel

    calc = runner.tool_registry.get("calculator")
    calc._info = calc.info.model_copy(update={"risk_level": RiskLevel.MEDIUM})

    paused = await runner.run(goal="What is 238 * 47?")
    assert paused.final_status is LoopFinalStatus.WAITING_FOR_CONFIRMATION

    # Approve the pending confirmation.
    pending = runner.confirmation_store.list_pending()
    assert len(pending) == 1
    confirm_id = pending[0].confirmation_id
    runner.confirmation_store.approve(confirm_id)

    # Resume the same loop.
    resumed = await runner.controller.resume_after_confirmation(
        runner.context, confirm_id
    )
    return paused, resumed


async def run_confirmation_rejection_demo() -> tuple[LoopResult, LoopResult]:
    """Demo 7 — confirmation required, then rejected (tool not executed).

    Returns (paused_result, resumed_result). The resumed result should NOT
    have executed the tool — verification fails and the loop fails safely.
    """
    runner = MathLoopRunner()
    await runner.setup(
        policy=DefaultToolPolicy(require_confirmation_for_medium=True)
    )
    from app.tools.interface import RiskLevel

    calc = runner.tool_registry.get("calculator")
    calc._info = calc.info.model_copy(update={"risk_level": RiskLevel.MEDIUM})

    paused = await runner.run(goal="What is 238 * 47?")
    assert paused.final_status is LoopFinalStatus.WAITING_FOR_CONFIRMATION

    pending = runner.confirmation_store.list_pending()
    assert len(pending) == 1
    confirm_id = pending[0].confirmation_id
    runner.confirmation_store.reject(confirm_id)

    resumed = await runner.controller.resume_after_rejection(
        runner.context, confirm_id, reason="rejected by user"
    )
    return paused, resumed
