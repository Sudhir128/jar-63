"""Plan stage.

Determines the next concrete action. Phase 1 ships a deterministic planner
that maps the discovered context to a single execute step. Phase 2 adds an
optional LLM-backed planner (:class:`LLMPlanStage`) that produces a validated
structured plan and falls back to the deterministic planner on any failure.

Both planners are deliberately replaceable. The loop engine is unchanged: it
consumes :class:`PlanResult` / :class:`NextAction` regardless of the source.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.core.identifiers import generate_id
from app.core.logging import get_logger
from app.runtime.loop.loop_context import LoopContext
from app.runtime.loop.loop_state import (
    ActionType,
    LoopStatus,
    NextAction,
    PlanStep,
    StageStatus,
)
from app.runtime.loop.stages.base import LoopStage

logger = get_logger("loop.plan")

__all__ = ["PlanResult", "PlanStage", "DefaultPlanStage", "LLMPlanStage"]


class PlanResult(BaseModel):
    """Typed output of the Plan stage."""

    model_config = ConfigDict(frozen=True, extra="allow")

    plan_id: str = Field(default_factory=lambda: generate_id("plan"))
    steps: list[PlanStep] = Field(default_factory=list)
    current_step: PlanStep | None = None
    next_action: NextAction = Field(default_factory=NextAction)
    reasoning_metadata: dict[str, object] = Field(default_factory=dict)


class PlanStage(LoopStage):
    """Abstract plan stage contract."""

    name = "plan"

    async def run(self, context: LoopContext) -> LoopContext:
        result = await self.plan(context)
        context.update_state(
            current_stage=StageStatus.PLAN,
            next_action=result.next_action,
            status=LoopStatus.PLANNING,
            current_plan=result.model_dump(),
            current_step=result.current_step,
            # Reset per-iteration tool call counter at the start of each plan.
            tool_call_count_per_iteration=0,
        )
        context.task.metadata["plan"] = result.model_dump()
        return context

    async def plan(self, context: LoopContext) -> PlanResult:
        """Produce a typed plan result."""


class DefaultPlanStage(PlanStage):
    """Deterministic planner.

    Reads the discovery result (stashed on task metadata by Discover) and
    produces a single execute step targeting the discovered agent. If the
    agent is unknown to the registry, the plan refuses to execute it — the
    loop never executes an unknown agent.
    """

    async def plan(self, context: LoopContext) -> PlanResult:
        discovery = context.task.metadata.get("discovery") or {}
        agent_id = discovery.get("agent_id")
        goal = context.state.goal

        if agent_id and context.agent_registry.exists(agent_id):
            step = PlanStep(
                description=f"Execute agent '{agent_id}' for goal: {goal}",
                action_type=ActionType.EXECUTE_AGENT,
                target=agent_id,
            )
            action = NextAction(
                action_type=ActionType.EXECUTE_AGENT,
                target=agent_id,
                description=f"Execute agent '{agent_id}'.",
            )
        else:
            # No resolvable agent — do not execute unknown agents.
            step = PlanStep(
                description="No agent available to satisfy the goal.",
                action_type=ActionType.NONE,
            )
            action = NextAction(
                action_type=ActionType.NONE,
                description="No agent available; nothing to execute.",
            )
        return PlanResult(
            steps=[step],
            current_step=step,
            next_action=action,
            reasoning_metadata={"planner": "deterministic", "agent_id": agent_id},
        )


class LLMPlanStage(PlanStage):
    """LLM-backed plan stage (Phase 2).

    Delegates planning to an :class:`~app.llm.planner.LLMPlanner`, which uses
    an LLM to produce a validated structured plan and falls back to the
    deterministic planner on any failure. The stage itself only wires the
    planner into the loop; all LLM logic lives in the planner.
    """

    def __init__(self, planner: object) -> None:
        # Typed loosely to avoid an import cycle; the planner exposes
        # ``async plan(context) -> PlanResult``.
        self._planner = planner

    async def plan(self, context: LoopContext) -> PlanResult:
        return await self._planner.plan(context)  # type: ignore[attr-defined]
