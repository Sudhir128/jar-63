"""Phase 3 demonstration workflows: LLM-powered loop + tool execution.

These prove the LLM-powered loop works end-to-end with the real
:class:`LoopController`. Each demo wires concrete tools, a stub/mock LLM
planner, and verifiers through the loop. Results are not faked.

* Demo 1 — LLM + Calculator workflow (calculate 238 * 47)
* Demo 2 — LLM + Time workflow (what time is it?)
* Demo 3 — Tool failure workflow (invalid expression)
* Demo 4 — Policy denial workflow (denied tool)
* Demo 5 — Confirmation workflow (medium-risk tool requiring confirmation)
* Demo 6 — Loop iteration after verification failure
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.agents.registry import AgentRegistry
from app.events import EventBus, InMemoryEventBus
from app.llm.planner import LLMPlanner
from app.llm.registry import ModelRegistry, ProviderRegistry
from app.llm.router import ModelRouter
from app.runtime.loop.loop_context import LoopContext
from app.runtime.loop.loop_controller import LoopController
from app.runtime.loop.loop_policy import LoopPolicy
from app.runtime.loop.loop_result import LoopResult
from app.runtime.loop.loop_state import LoopState
from app.runtime.loop.stages import (
    DefaultDiscoverStage,
    DefaultIterateStage,
    DefaultPlanStage,
    DefaultVerifyStage,
)
from app.runtime.loop.stages.execute import DefaultExecuteStage
from app.runtime.loop.stages.plan import LLMPlanStage
from app.runtime.loop.verification import CallableVerifier, ExactMatchVerifier
from app.runtime.models import Task
from app.tools.confirmation import ConfirmationStore
from app.tools.executor import ToolExecutor
from app.tools.impl import CalculatorTool, EchoTool, HealthTool, TimeTool
from app.tools.interface import (
    RiskLevel,
    ToolCategory,
    ToolContext,
    ToolInfo,
    ToolInterface,
    ToolResult,
)
from app.tools.policy import DefaultToolPolicy
from app.tools.registry import ToolRegistry

__all__ = [
    "Phase3Runner",
    "MockMediumRiskTool",
    "run_calculator_workflow",
    "run_time_workflow",
    "run_tool_failure_workflow",
    "run_policy_denial_workflow",
    "run_confirmation_workflow",
    "run_iteration_workflow",
    "run_llm_planner_workflow",
]


@dataclass
class Phase3Runner:
    """Wires registries, tool executor, event bus, and a LoopController."""

    agent_registry: AgentRegistry = field(default_factory=AgentRegistry)
    tool_registry: ToolRegistry = field(default_factory=ToolRegistry)
    event_bus: EventBus = field(default_factory=InMemoryEventBus)
    confirmation_store: ConfirmationStore = field(default_factory=ConfirmationStore)
    tool_policy: DefaultToolPolicy = field(default_factory=DefaultToolPolicy)
    _executor: ToolExecutor | None = None

    @property
    def tool_executor(self) -> ToolExecutor:
        if self._executor is None:
            self._executor = ToolExecutor(
                registry=self.tool_registry,
                policy=self.tool_policy,
                confirmation_store=self.confirmation_store,
                event_bus=self.event_bus,
            )
        return self._executor

    async def register_tools(self, tools: list[ToolInterface]) -> None:
        for t in tools:
            await self.tool_registry.register(t)

    def make_controller(
        self,
        *,
        plan_stage: Any | None = None,
        verifier: Any | None = None,
        max_iterations: int = 5,
        expected_output: Any = None,
    ) -> tuple[LoopController, LoopPolicy]:
        policy = LoopPolicy(max_iterations=max_iterations)
        controller = LoopController(
            discover_stage=DefaultDiscoverStage(),
            plan_stage=plan_stage or DefaultPlanStage(),
            execute_stage=DefaultExecuteStage(tool_executor=self.tool_executor),
            verify_stage=DefaultVerifyStage(verifier=verifier),
            iterate_stage=DefaultIterateStage(),
        )
        return controller, policy

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


def _make_task(goal: str, agent_id: str | None = None) -> Task:
    return Task(input=goal, agent_id=agent_id, metadata={})


def _calc_tool_step_payload(expression: str) -> dict:
    return {
        "goal": f"Calculate {expression}",
        "steps": [
            {
                "description": f"Use the calculator tool to evaluate {expression}.",
                "capability": "arithmetic",
                "tool": "calculator",
                "tool_arguments": {"expression": expression},
                "expected_output": "the numeric result",
                "verification_requirements": ["result is a number"],
            }
        ],
        "assumptions": ["integer arithmetic"],
        "success_criteria": ["result matches expected value"],
    }


def _time_tool_step_payload() -> dict:
    return {
        "goal": "What time is it?",
        "steps": [
            {
                "description": "Use the time tool to get the current time.",
                "capability": "time",
                "tool": "time",
                "tool_arguments": {},
                "expected_output": "current UTC time",
                "verification_requirements": ["time is present"],
            }
        ],
        "assumptions": [],
        "success_criteria": ["time is returned"],
    }


# ---------------------------------------------------------------------------
# Stub LLM client that returns a pre-built plan
# ---------------------------------------------------------------------------


def _make_stub_planner(plan_payload: dict) -> LLMPlanner:
    """Build an LLMPlanner backed by a stub client returning plan_payload."""
    from tests.llm_helpers import StubLLMClient, local_coding_model, make_llm_settings

    client = StubLLMClient("ollama", structured_payload=plan_payload)
    mr = ModelRegistry()
    mr.register(local_coding_model())
    pr = ProviderRegistry()
    pr.register("ollama", client)
    router = ModelRouter(model_registry=mr, provider_registry=pr, settings=make_llm_settings())
    return LLMPlanner(
        router=router,
        settings=make_llm_settings(),
        event_bus=InMemoryEventBus(),
        client_override=client,
    )


# ---------------------------------------------------------------------------
# Demo 1: Calculator workflow
# ---------------------------------------------------------------------------


async def run_calculator_workflow() -> LoopResult:
    """LLM plans a calculator tool call → execute → observe → verify → success."""
    runner = Phase3Runner()
    await runner.register_tools([CalculatorTool(), TimeTool(), HealthTool(), EchoTool()])

    expression = "238 * 47"
    expected = 11186
    planner = _make_stub_planner(_calc_tool_step_payload(expression))
    controller, policy = runner.make_controller(
        plan_stage=LLMPlanStage(planner),
        verifier=CallableVerifier(_make_result_checker(expected), check_name="calculator_result"),
        max_iterations=3,
        expected_output=expected,
    )
    task = _make_task(f"Calculate {expression}")
    state = LoopState(
        task_id=task.task_id,
        goal=f"Calculate {expression}",
        success_criteria=[f"expected:{expected}"],
        max_iterations=3,
    )
    context = runner.make_context(task, state, policy, expected_output=expected)
    return await controller.run(context)


def _make_result_checker(expected: Any):
    """A verifier predicate that extracts the numeric result from tool output."""

    def check(output: Any) -> bool:
        if isinstance(output, dict) and "result" in output:
            return output["result"] == expected
        return False

    return check


# ---------------------------------------------------------------------------
# Demo 2: Time workflow
# ---------------------------------------------------------------------------


async def run_time_workflow() -> LoopResult:
    """LLM plans a time tool call → execute → observe → verify → success."""
    runner = Phase3Runner()
    await runner.register_tools([CalculatorTool(), TimeTool(), HealthTool(), EchoTool()])

    planner = _make_stub_planner(_time_tool_step_payload())
    controller, policy = runner.make_controller(
        plan_stage=LLMPlanStage(planner),
        verifier=CallableVerifier(_time_check, check_name="time_returned"),
        max_iterations=3,
    )
    task = _make_task("What time is it?")
    state = LoopState(
        task_id=task.task_id,
        goal="What time is it?",
        success_criteria=["time is returned"],
        max_iterations=3,
    )
    context = runner.make_context(task, state, policy)
    return await controller.run(context)


def _time_check(output: Any) -> bool:
    return isinstance(output, dict) and "utc_iso" in output


# ---------------------------------------------------------------------------
# Demo 3: Tool failure workflow
# ---------------------------------------------------------------------------


async def run_tool_failure_workflow() -> LoopResult:
    """Calculator receives an invalid expression → tool fails → no false success."""
    runner = Phase3Runner()
    await runner.register_tools([CalculatorTool(), TimeTool(), HealthTool(), EchoTool()])

    expression = "1 / 0"
    planner = _make_stub_planner(_calc_tool_step_payload(expression))
    controller, policy = runner.make_controller(
        plan_stage=LLMPlanStage(planner),
        verifier=ExactMatchVerifier(check_name="should_not_reach"),
        max_iterations=2,
    )
    task = _make_task(f"Calculate {expression}")
    state = LoopState(
        task_id=task.task_id,
        goal=f"Calculate {expression}",
        success_criteria=["expected:undefined"],
        max_iterations=2,
    )
    context = runner.make_context(task, state, policy)
    return await controller.run(context)


# ---------------------------------------------------------------------------
# Demo 4: Policy denial workflow
# ---------------------------------------------------------------------------


class MockMediumRiskTool(ToolInterface):
    """A harmless mock medium-risk tool used to demonstrate policy denial."""

    def __init__(self, name: str = "mock_medium") -> None:
        self._info = ToolInfo(
            tool_id=name,
            name=name,
            description="A harmless mock tool (medium risk for demo).",
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
            output="mock medium result",
        )


async def run_policy_denial_workflow() -> LoopResult:
    """LLM requests a denied tool → policy DENY → tool not executed → no false success."""
    runner = Phase3Runner()
    runner.tool_policy = DefaultToolPolicy(deny={"mock_medium"})
    runner._executor = None  # force rebuild with new policy
    await runner.register_tools([CalculatorTool(), MockMediumRiskTool()])

    plan_payload = {
        "goal": "Use the mock_medium tool.",
        "steps": [
            {
                "description": "Call the mock_medium tool.",
                "capability": "custom",
                "tool": "mock_medium",
                "tool_arguments": {},
                "expected_output": "mock medium result",
                "verification_requirements": ["result is returned"],
            }
        ],
        "assumptions": [],
        "success_criteria": ["mock_medium result returned"],
    }
    planner = _make_stub_planner(plan_payload)
    controller, policy = runner.make_controller(
        plan_stage=LLMPlanStage(planner),
        verifier=ExactMatchVerifier(check_name="should_not_match"),
        max_iterations=2,
    )
    task = _make_task("Use the mock_medium tool.")
    state = LoopState(
        task_id=task.task_id,
        goal="Use the mock_medium tool.",
        success_criteria=["expected:never"],
        max_iterations=2,
    )
    context = runner.make_context(task, state, policy)
    return await controller.run(context)


# ---------------------------------------------------------------------------
# Demo 5: Confirmation workflow
# ---------------------------------------------------------------------------


async def run_confirmation_workflow() -> tuple[LoopResult, Phase3Runner, str | None]:
    """LLM requests a HIGH-risk tool → policy CONFIRM → loop pauses → approve → execute.

    Returns (result, runner, confirmation_id) so the caller can inspect state.
    The first run pauses (returns FAILED since confirmation is pending). The
    caller approves via the confirmation store, then resumes.
    """
    runner = Phase3Runner()
    await runner.register_tools([CalculatorTool(), MockMediumRiskTool()])

    plan_payload = {
        "goal": "Use the mock_medium tool with confirmation.",
        "steps": [
            {
                "description": "Call the mock_medium tool (requires confirmation).",
                "capability": "custom",
                "tool": "mock_medium",
                "tool_arguments": {},
                "expected_output": "mock medium result",
                "verification_requirements": ["result is returned"],
            }
        ],
        "assumptions": [],
        "success_criteria": ["mock_medium result returned"],
    }
    planner = _make_stub_planner(plan_payload)
    controller, policy = runner.make_controller(
        plan_stage=LLMPlanStage(planner),
        verifier=CallableVerifier(
            lambda o: isinstance(o, str) and "mock" in o, check_name="mock_result"
        ),
        max_iterations=3,
    )
    task = _make_task("Use the mock_medium tool with confirmation.")
    state = LoopState(
        task_id=task.task_id,
        goal="Use the mock_medium tool with confirmation.",
        success_criteria=["mock_medium result returned"],
        max_iterations=3,
    )
    context = runner.make_context(task, state, policy)
    result = await controller.run(context)
    # Find the confirmation id from state.
    confirm_id = None
    if context.state.confirmation_request:
        confirm_id = context.state.confirmation_request.get("confirmation_id")
    return result, runner, confirm_id


async def resume_confirmation_workflow(runner: Phase3Runner, confirmation_id: str) -> LoopResult:
    """Approve the pending confirmation and execute the tool via the executor."""
    runner.confirmation_store.approve(confirmation_id)
    outcome = await runner.tool_executor.execute_confirmed(confirmation_id, task_id="resume")
    from app.runtime.loop.loop_result import LoopFinalStatus

    return LoopResult(
        loop_id="resume",
        task_id="resume",
        final_status=LoopFinalStatus.SUCCESS
        if outcome.result and outcome.result.success
        else LoopFinalStatus.FAILED,
        success=bool(outcome.result and outcome.result.success),
        final_response=outcome.result.output if outcome.result else None,
        iterations_used=1,
    )


# ---------------------------------------------------------------------------
# Demo 6: Loop iteration after verification failure
# ---------------------------------------------------------------------------


async def run_iteration_workflow() -> LoopResult:
    """First plan uses a wrong expression → verify fails → iterate → new plan → success.

    Uses a scripted planner that returns a failing plan first, then a correct one.
    """
    runner = Phase3Runner()
    await runner.register_tools([CalculatorTool(), TimeTool()])

    class _ScriptedPlanner:
        def __init__(self) -> None:
            self._call = 0

        async def plan(self, context: LoopContext) -> Any:
            from app.runtime.loop.loop_state import ActionType, NextAction, PlanStep
            from app.runtime.loop.stages.plan import PlanResult

            self._call += 1
            # Wrong on first call (won't match expected 11186); correct on second.
            expr = "1 + 1" if self._call == 1 else "238 * 47"
            step = PlanStep(
                description=f"Calculate {expr}",
                action_type=ActionType.EXECUTE_TOOL,
                target="calculator",
                parameters={"arguments": {"expression": expr}},
            )
            return PlanResult(
                steps=[step],
                current_step=step,
                next_action=NextAction(
                    action_type=ActionType.EXECUTE_TOOL,
                    target="calculator",
                    description=f"Calculate {expr}",
                    parameters={"arguments": {"expression": expr}},
                ),
                reasoning_metadata={"planner": "scripted", "call": self._call},
            )

    controller, policy = runner.make_controller(
        plan_stage=LLMPlanStage(_ScriptedPlanner()),
        verifier=CallableVerifier(_make_result_checker(11186), check_name="equals_11186"),
        max_iterations=3,
        expected_output=11186,
    )
    task = _make_task("Calculate 238 * 47")
    state = LoopState(
        task_id=task.task_id,
        goal="Calculate 238 * 47",
        success_criteria=["expected:11186"],
        max_iterations=3,
    )
    context = runner.make_context(task, state, policy, expected_output=11186)
    return await controller.run(context)


# ---------------------------------------------------------------------------
# Demo 7: LLM planner (full end-to-end with stub LLM)
# ---------------------------------------------------------------------------


async def run_llm_planner_workflow() -> LoopResult:
    """Full LLM planner workflow: plan → validate → execute → verify → success."""
    return await run_calculator_workflow()
