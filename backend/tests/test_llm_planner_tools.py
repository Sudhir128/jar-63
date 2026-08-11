"""Tests for LLM planner tool-calling support and plan validation."""

from __future__ import annotations

from app.events import InMemoryEventBus
from app.llm.plan_schema import LLMPlan, LLMPlanStep
from app.llm.planner import LLMPlanner
from app.llm.registry import ModelRegistry, ProviderRegistry
from app.llm.router import ModelRouter
from app.runtime.loop.loop_context import LoopContext
from app.runtime.loop.loop_policy import LoopPolicy
from app.runtime.loop.loop_state import LoopState
from app.runtime.models import Task
from app.tools.impl import CalculatorTool
from app.tools.registry import ToolRegistry
from tests.llm_helpers import StubLLMClient, local_coding_model, make_llm_settings


class StubAgentRegistry:
    def __init__(self, known: set[str] | None = None) -> None:
        self._known = known or set()

    def exists(self, agent_id: str) -> bool:
        return agent_id in self._known


def _tool_context(*, goal="Calculate 10 * 10") -> LoopContext:
    state = LoopState(task_id="t1", goal=goal, success_criteria=["expected:100"])
    task = Task(task_id="t1", metadata={"discovery": {"available_tools": ["calculator"]}})
    return LoopContext(
        state=state,
        task=task,
        agent_registry=StubAgentRegistry(),
        tool_registry=ToolRegistry(),
        event_bus=InMemoryEventBus(),
        policy=LoopPolicy(),
    )


def _router(client: StubLLMClient) -> ModelRouter:
    mr = ModelRegistry()
    mr.register(local_coding_model())
    pr = ProviderRegistry()
    pr.register("ollama", client)
    return ModelRouter(model_registry=mr, provider_registry=pr, settings=make_llm_settings())


def _tool_plan_payload(tool="calculator", args=None) -> dict:
    return {
        "goal": "Calculate 10 * 10",
        "steps": [
            {
                "description": "Use the calculator to compute 10 * 10.",
                "capability": "arithmetic",
                "tool": tool,
                "tool_arguments": args or {"expression": "10 * 10"},
                "expected_output": "100",
                "verification_requirements": ["result == 100"],
            }
        ],
        "assumptions": [],
        "success_criteria": ["result == 100"],
    }


# --- Plan schema: tool steps ---


def test_plan_step_with_tool() -> None:
    step = LLMPlanStep(description="calc", tool="calculator", tool_arguments={"expression": "1+1"})
    assert step.is_tool_step
    assert step.tool == "calculator"


def test_plan_step_without_tool() -> None:
    step = LLMPlanStep(description="agent step", agent_id="math.agent")
    assert not step.is_tool_step


def test_plan_with_tool_step_validates() -> None:
    plan = LLMPlan(
        goal="Calculate",
        steps=[LLMPlanStep(description="calc", tool="calculator", tool_arguments={})],
        success_criteria=["result"],
    )
    assert plan.first_step.is_tool_step


# --- Planner: tool plan translation ---


async def test_planner_translates_tool_plan() -> None:
    client = StubLLMClient("ollama", structured_payload=_tool_plan_payload())
    planner = LLMPlanner(
        router=_router(client), settings=make_llm_settings(), client_override=client
    )
    ctx = _tool_context()
    # Register the calculator so it resolves.
    await ctx.tool_registry.register(CalculatorTool())
    result = await planner.plan(ctx)
    assert result.reasoning_metadata["planner"] == "llm"
    assert result.next_action.action_type.value == "execute_tool"
    assert result.next_action.target == "calculator"
    assert result.next_action.parameters["arguments"]["expression"] == "10 * 10"


async def test_planner_unknown_tool_becomes_noop() -> None:
    client = StubLLMClient("ollama", structured_payload=_tool_plan_payload(tool="nonexistent"))
    planner = LLMPlanner(
        router=_router(client), settings=make_llm_settings(), client_override=client
    )
    ctx = _tool_context()
    await ctx.tool_registry.register(CalculatorTool())
    result = await planner.plan(ctx)
    assert result.next_action.action_type.value == "none"
    assert "unknown tool" in result.next_action.description.lower()


async def test_planner_prompt_includes_available_tools() -> None:
    """The planner prompt should list available tools so the LLM can choose them."""
    client = StubLLMClient("ollama", structured_payload=_tool_plan_payload())
    planner = LLMPlanner(
        router=_router(client), settings=make_llm_settings(), client_override=client
    )
    ctx = _tool_context()
    selection = await planner._select_model(ctx)  # noqa: SLF001
    messages = planner._build_prompt(ctx, selection)  # noqa: SLF001
    user_msg = messages[-1].content
    assert "calculator" in user_msg


# --- Planner fallback ---


async def test_planner_falls_back_on_invalid_tool_plan() -> None:
    """If the LLM produces an invalid plan, the planner falls back."""
    # Plan with no success criteria → ValidationError.
    bad_payload = {"goal": "test", "steps": [], "assumptions": [], "success_criteria": []}
    client = StubLLMClient("ollama", structured_payload=bad_payload)
    planner = LLMPlanner(
        router=_router(client), settings=make_llm_settings(), client_override=client
    )
    ctx = _tool_context()
    result = await planner.plan(ctx)
    assert result.reasoning_metadata["planner"] in ("deterministic_fallback", "deterministic")
