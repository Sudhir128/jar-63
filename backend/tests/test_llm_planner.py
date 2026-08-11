"""Tests for the LLM planner integration and deterministic fallback."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.events import EventType
from app.llm.errors import LLMError
from app.llm.plan_schema import LLMPlan, LLMPlanStep
from app.llm.planner import LLMPlanner, PlannerFallbackReason
from app.llm.registry import ModelRegistry, ProviderRegistry
from app.llm.router import ModelRouter
from app.runtime.loop.loop_context import LoopContext
from app.runtime.loop.loop_policy import LoopPolicy
from app.runtime.loop.loop_state import LoopState
from app.runtime.loop.stages.plan import LLMPlanStage
from app.runtime.models import Task
from app.tools.registry import ToolRegistry
from tests.llm_helpers import StubLLMClient, local_coding_model, make_llm_settings


class StubAgentRegistry:
    def __init__(self, known: set[str] | None = None) -> None:
        self._known = known or set()

    def exists(self, agent_id: str) -> bool:
        return agent_id in self._known


def _context(*, goal="Calculate 25 * 4", agent_id="math.agent", known_agents=None) -> LoopContext:
    from app.events import InMemoryEventBus

    state = LoopState(task_id="t1", goal=goal, success_criteria=["expected:100"])
    task = Task(task_id="t1", agent_id=agent_id, metadata={"discovery": {"agent_id": agent_id}})
    registry_known = known_agents if known_agents is not None else {agent_id}
    return LoopContext(
        state=state,
        task=task,
        agent_registry=StubAgentRegistry(registry_known),  # type: ignore[arg-type]
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


def _valid_plan_payload(agent_id="math.agent") -> dict:
    return {
        "goal": "Calculate 25 * 4",
        "steps": [
            {
                "description": "Compute 25 * 4 using the math agent.",
                "capability": "math",
                "agent_id": agent_id,
                "expected_output": "100",
                "verification_requirements": ["result == 100"],
            }
        ],
        "assumptions": ["integer arithmetic"],
        "success_criteria": ["result == 100"],
    }


# --- Plan schema validation ---


def test_plan_step_requires_description() -> None:
    with pytest.raises(ValidationError):
        LLMPlanStep(description="")


def test_plan_requires_success_criteria() -> None:
    with pytest.raises(ValidationError, match="success criterion"):
        LLMPlan(goal="x", steps=[LLMPlanStep(description="do something")])


def test_plan_requires_goal() -> None:
    with pytest.raises(ValidationError, match="goal"):
        LLMPlan(goal="", success_criteria=["x"])


def test_plan_first_step() -> None:
    plan = LLMPlan(
        goal="x",
        steps=[LLMPlanStep(description="a"), LLMPlanStep(description="b")],
        success_criteria=["done"],
    )
    assert plan.first_step.description == "a"


# --- LLM planner success ---


async def test_planner_produces_llm_plan() -> None:
    client = StubLLMClient("ollama", structured_payload=_valid_plan_payload())
    planner = LLMPlanner(
        router=_router(client), settings=make_llm_settings(), client_override=client
    )
    result = await planner.plan(_context())
    assert result.reasoning_metadata["planner"] == "llm"
    assert result.next_action.target == "math.agent"
    assert "success_criteria" in result.reasoning_metadata


async def test_planner_unknown_agent_becomes_noop() -> None:
    client = StubLLMClient("ollama", structured_payload=_valid_plan_payload(agent_id="nonexistent"))
    planner = LLMPlanner(
        router=_router(client), settings=make_llm_settings(), client_override=client
    )
    # Context knows no agents.
    ctx = _context(agent_id="nonexistent", known_agents=set())
    result = await planner.plan(ctx)
    # Falls into no-op because the agent is unknown; planner still marked llm.
    assert result.reasoning_metadata["planner"] == "llm"
    from app.runtime.loop.loop_state import ActionType

    assert result.next_action.action_type is ActionType.NONE


async def test_planner_llm_does_not_execute_directly() -> None:
    """The planner only produces a plan; it never executes tools/agents."""
    client = StubLLMClient("ollama", structured_payload=_valid_plan_payload())
    planner = LLMPlanner(
        router=_router(client), settings=make_llm_settings(), client_override=client
    )
    result = await planner.plan(_context())
    # The result is a PlanResult with a next_action describing what to do;
    # nothing was executed. There is no execution result on the state.
    assert result.next_action.target == "math.agent"
    # No execution_results populated by the planner.
    assert result.next_action.action_type.value in ("execute_agent", "none")


# --- Deterministic fallback ---


async def test_planner_falls_back_when_llm_disabled() -> None:
    client = StubLLMClient("ollama", structured_payload=_valid_plan_payload())
    settings = make_llm_settings(enabled=False)
    planner = LLMPlanner(router=_router(client), settings=settings, client_override=client)
    result = await planner.plan(_context())
    assert result.reasoning_metadata["planner"] == "deterministic_fallback"
    assert result.reasoning_metadata["fallback_reason"] == PlannerFallbackReason.LLM_DISABLED


async def test_planner_falls_back_on_no_model() -> None:
    # Router with no models -> ModelUnavailableError -> fallback.
    client = StubLLMClient("ollama")
    mr = ModelRegistry()
    pr = ProviderRegistry()
    pr.register("ollama", client)
    router = ModelRouter(model_registry=mr, provider_registry=pr, settings=make_llm_settings())
    planner = LLMPlanner(router=router, settings=make_llm_settings(), client_override=None)
    result = await planner.plan(_context())
    assert result.reasoning_metadata["planner"] == "deterministic_fallback"
    assert result.reasoning_metadata["fallback_reason"] == PlannerFallbackReason.NO_MODEL


async def test_planner_falls_back_on_llm_error() -> None:
    client = StubLLMClient("ollama", raise_error=LLMError("boom"))
    planner = LLMPlanner(
        router=_router(client), settings=make_llm_settings(), client_override=client
    )
    result = await planner.plan(_context())
    assert result.reasoning_metadata["planner"] == "deterministic_fallback"
    assert result.reasoning_metadata["fallback_reason"] == PlannerFallbackReason.LLM_ERROR


async def test_planner_falls_back_on_invalid_output() -> None:
    # Return a payload that fails schema validation (missing success_criteria).
    client = StubLLMClient("ollama", structured_payload={"goal": "x", "steps": []})
    planner = LLMPlanner(
        router=_router(client), settings=make_llm_settings(), client_override=client
    )
    result = await planner.plan(_context())
    assert result.reasoning_metadata["planner"] == "deterministic_fallback"
    assert result.reasoning_metadata["fallback_reason"] in (
        PlannerFallbackReason.INVALID_OUTPUT,
        PlannerFallbackReason.EMPTY_PLAN,
    )


async def test_planner_falls_back_on_empty_plan() -> None:
    client = StubLLMClient(
        "ollama",
        structured_payload={"goal": "x", "steps": [], "success_criteria": ["done"]},
    )
    planner = LLMPlanner(
        router=_router(client), settings=make_llm_settings(), client_override=client
    )
    result = await planner.plan(_context())
    assert result.reasoning_metadata["fallback_reason"] == PlannerFallbackReason.EMPTY_PLAN


async def test_planner_publishes_fallback_event(captured_events) -> None:
    bus, events = captured_events
    client = StubLLMClient("ollama", structured_payload=_valid_plan_payload())
    settings = make_llm_settings(enabled=False)
    planner = LLMPlanner(
        router=_router(client), settings=settings, client_override=client, event_bus=bus
    )
    await planner.plan(_context())
    assert any(e.event_type is EventType.MODEL_FALLBACK for e in events)


async def test_planner_no_prompt_content_in_events(captured_events) -> None:
    bus, events = captured_events
    client = StubLLMClient("ollama", structured_payload=_valid_plan_payload())
    planner = LLMPlanner(
        router=_router(client), settings=make_llm_settings(), client_override=client, event_bus=bus
    )
    await planner.plan(_context())
    import json

    for e in events:
        blob = json.dumps(e.payload)
        # The goal text must not appear in events.
        assert "Calculate 25 * 4" not in blob


# --- LLMPlanStage integration ---


async def test_llm_plan_stage_delegates_to_planner() -> None:
    client = StubLLMClient("ollama", structured_payload=_valid_plan_payload())
    planner = LLMPlanner(
        router=_router(client), settings=make_llm_settings(), client_override=client
    )
    stage = LLMPlanStage(planner)
    ctx = _context()
    await stage.run(ctx)
    plan_meta = ctx.task.metadata["plan"]
    assert plan_meta["reasoning_metadata"]["planner"] == "llm"
    assert ctx.state.next_action.target == "math.agent"
