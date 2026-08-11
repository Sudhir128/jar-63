"""LangGraph adapter for the Universal Loop Engine.

This module is the **boundary** between the domain loop model and LangGraph.
LangGraph is used as the state-machine runner; all loop logic lives in
:class:`LoopController` and its stages. The adapter translates between
LangGraph's graph state and the domain :class:`LoopState`.

If LangGraph is ever replaced, only this file changes — the controller,
stages, verification, and stop conditions remain intact.

Graph::

    START → discover → plan → execute → verify → decide
        decide ├── success/cancelled/max/timeout/failure → END
                └── iterate → plan   (loop back)
"""

from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.core.logging import get_logger
from app.runtime.loop.conditions import (
    CancellationCondition,
    FailureCondition,
    MaxIterationsCondition,
    StopDecision,
    SuccessCondition,
    TimeoutCondition,
)
from app.runtime.loop.loop_context import LoopContext
from app.runtime.loop.loop_controller import LoopController
from app.runtime.loop.loop_result import LoopFinalStatus, LoopResult
from app.runtime.loop.loop_state import StageStatus

logger = get_logger("loop.graph")

__all__ = ["LoopGraphState", "LoopGraphAdapter"]


class LoopGraphState(TypedDict, total=False):
    """LangGraph state carrying only the loop_id; the controller holds state.

    Keeping the graph state thin avoids duplicating :class:`LoopState`. The
    adapter looks up the live context by ``loop_id``.
    """

    loop_id: str
    decision: str
    terminated: bool


class LoopGraphAdapter:
    """Wraps a :class:`LoopController` in a LangGraph state machine.

    The adapter does not duplicate loop logic. Each node delegates to the
    controller's stages or stop-condition evaluation. The controller remains
    the single source of truth.
    """

    def __init__(self, controller: LoopController) -> None:
        self.controller = controller
        self._graph: CompiledStateGraph | None = None

    @property
    def graph(self) -> CompiledStateGraph:
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph

    def _build_graph(self) -> CompiledStateGraph:
        builder: StateGraph = StateGraph(LoopGraphState)
        builder.add_node("discover", self._discover_node)
        builder.add_node("plan", self._plan_node)
        builder.add_node("execute", self._execute_node)
        builder.add_node("verify", self._verify_node)
        builder.add_node("decide", self._decide_node)
        builder.add_node("iterate", self._iterate_node)

        builder.add_edge(START, "discover")
        builder.add_edge("discover", "plan")
        builder.add_conditional_edges(
            "plan",
            self._after_plan,
            {"execute": "execute", "end": END},
        )
        builder.add_edge("execute", "verify")
        builder.add_conditional_edges(
            "verify",
            self._after_verify,
            {"decide": "decide", "end": END},
        )
        builder.add_conditional_edges(
            "decide",
            self._after_decide,
            {"end": END, "iterate": "iterate"},
        )
        builder.add_edge("iterate", "plan")
        return builder.compile()

    # --- nodes (delegate to the controller's stages) ---
    async def _discover_node(self, state: LoopGraphState) -> LoopGraphState:
        ctx = self.controller.context
        await self.controller.discover_stage.run(ctx)
        return {"loop_id": ctx.loop_id}

    async def _plan_node(self, state: LoopGraphState) -> LoopGraphState:
        ctx = self.controller.context
        # Begin a new iteration at the start of each plan cycle.
        if ctx.state.iteration_count == 0 or ctx.state.current_stage is StageStatus.ITERATE:
            ctx.state = ctx.state.begin_iteration()
        await self.controller.plan_stage.run(ctx)
        return {"loop_id": ctx.loop_id}

    async def _execute_node(self, state: LoopGraphState) -> LoopGraphState:
        ctx = self.controller.context
        await self.controller.execute_stage.run(ctx)
        return {"loop_id": ctx.loop_id}

    async def _verify_node(self, state: LoopGraphState) -> LoopGraphState:
        ctx = self.controller.context
        await self.controller.verify_stage.run(ctx)
        return {"loop_id": ctx.loop_id}

    async def _decide_node(self, state: LoopGraphState) -> LoopGraphState:
        decision = self._evaluate()
        return {
            "loop_id": self.controller.context.loop_id,
            "decision": decision.status.value if decision.status else "continue",
        }

    async def _iterate_node(self, state: LoopGraphState) -> LoopGraphState:
        ctx = self.controller.context
        await self.controller.iterate_stage.run(ctx)
        return {"loop_id": ctx.loop_id}

    # --- routing ---
    def _after_plan(self, state: LoopGraphState) -> Literal["execute", "end"]:
        ctx = self.controller.context
        action = ctx.state.next_action
        if action is None or action.action_type.value == "none":
            return "end"
        return "execute"

    def _after_verify(self, state: LoopGraphState) -> Literal["decide", "end"]:
        if self.controller.context.state.cancel_requested:
            return "end"
        return "decide"

    def _after_decide(self, state: LoopGraphState) -> Literal["end", "iterate"]:
        decision = self._evaluate()
        return "end" if decision.should_stop else "iterate"

    # --- helpers ---
    def _evaluate(self) -> StopDecision:
        conditions = [
            CancellationCondition(),
            SuccessCondition(),
            MaxIterationsCondition(),
            TimeoutCondition(),
            FailureCondition(),
        ]
        for cond in conditions:
            decision = cond.evaluate(self.controller.context)
            if decision.should_stop:
                return decision
        return StopDecision.continue_loop()

    async def run(self, context: LoopContext, *, recursion_limit: int = 100) -> LoopResult:
        """Execute the loop via the LangGraph state machine."""
        self.controller.initialize(context)
        ctx = self.controller.context
        from app.events import Event, EventType

        await ctx.event_bus.publish(
            Event.create(
                EventType.LOOP_STARTED,
                task_id=ctx.task_id,
                session_id=ctx.session_id,
                payload={
                    "goal": ctx.state.goal,
                    "max_iterations": ctx.state.max_iterations,
                    "engine": "langgraph",
                },
                metadata={"loop_id": ctx.loop_id, "message": "Starting work on the objective."},
            )
        )
        # Begin the first iteration before the graph runs.
        ctx.state = ctx.state.begin_iteration()

        terminated_status: LoopFinalStatus | None = None
        stopped_reason: str | None = None
        try:
            await self.graph.ainvoke(
                {"loop_id": ctx.loop_id},
                {"recursion_limit": recursion_limit},
            )
        except Exception as exc:  # noqa: BLE001
            logger.bind(
                event="loop.graph.error", loop_id=ctx.loop_id, error=type(exc).__name__
            ).warning("LangGraph loop ended early: {}", str(exc))
            terminated_status = LoopFinalStatus.FAILED
            stopped_reason = str(exc)

        # Determine the terminal status from the last stop decision.
        if terminated_status is None:
            decision = self._evaluate()
            if decision.should_stop and decision.status is not None:
                terminated_status = decision.status
                stopped_reason = decision.reason
            else:
                # Graph exited without a stop condition firing (e.g. recursion
                # limit). Treat as max iterations reached for safety.
                terminated_status = LoopFinalStatus.MAX_ITERATIONS_REACHED
                stopped_reason = "LangGraph recursion limit reached."

        failure_reason = (
            stopped_reason if terminated_status is not LoopFinalStatus.SUCCESS else None
        )
        # Reuse the controller's finalize path for consistent event publishing.
        return await self.controller._finalize(
            terminated_status,
            failure_reason=failure_reason,
            stopped_reason=stopped_reason,
        )
