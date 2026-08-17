"""LoopController: the Universal Loop Engine orchestrator.

The controller is **infrastructure**, not intelligence. It runs the stage
sequence (DISCOVER → PLAN → EXECUTE → VERIFY → DECIDE → ITERATE), maintains
state, evaluates stop conditions, publishes events, and enforces the policy.
Stage implementations provide the behavior; the controller never contains
LLM-specific reasoning.

Lifecycle::

    DISCOVER → PLAN → EXECUTE → VERIFY → DECIDE
        SUCCESS        → DONE
        CANCELLED      → DONE
        MAX LIMIT      → DONE
        FAILURE        → ITERATE → PLAN   (bounded by max_iterations)

The loop is never uncontrolled: every run terminates with a
:class:`LoopFinalStatus`.
"""

from __future__ import annotations

from typing import Any

from app.core.identifiers import utc_now
from app.core.logging import get_logger
from app.events import Event, EventType
from app.runtime.loop.conditions import (
    StopCondition,
    StopDecision,
    default_stop_conditions,
)
from app.runtime.loop.loop_context import LoopContext
from app.runtime.loop.loop_errors import LoopCancelledError, LoopStageError
from app.runtime.loop.loop_result import LoopFinalStatus, LoopResult
from app.runtime.loop.loop_state import (
    IterationRecord,
    LoopState,
    LoopStatus,
    StageStatus,
)
from app.runtime.loop.stages import (
    DefaultDiscoverStage,
    DefaultExecuteStage,
    DefaultIterateStage,
    DefaultPlanStage,
    DefaultVerifyStage,
    DiscoverStage,
    ExecuteStage,
    IterateStage,
    PlanStage,
    VerifyStage,
)
from app.runtime.models import Task

logger = get_logger("loop.controller")

__all__ = ["LoopController"]


# Human-readable messages for each stage/event — usable by the future Voice Agent.
_STAGE_MESSAGES = {
    StageStatus.DISCOVER: "Understanding the task and available context.",
    StageStatus.PLAN: "Planning the next action.",
    StageStatus.EXECUTE: "Executing the planned action.",
    StageStatus.VERIFY: "Verifying the result against the objective.",
    StageStatus.ITERATE: "Verification failed. I am adjusting the plan.",
    StageStatus.DECIDE: "Deciding whether the objective is achieved.",
}


class LoopController:
    """Orchestrates the Universal Loop Engine for a single task.

    The controller owns no agent/tool/LLM logic. All collaborators are
    injected via :class:`LoopContext`.
    """

    def __init__(
        self,
        *,
        discover_stage: DiscoverStage | None = None,
        plan_stage: PlanStage | None = None,
        execute_stage: ExecuteStage | None = None,
        verify_stage: VerifyStage | None = None,
        iterate_stage: IterateStage | None = None,
        stop_conditions: list[StopCondition] | None = None,
    ) -> None:
        self.discover_stage = discover_stage or DefaultDiscoverStage()
        self.plan_stage = plan_stage or DefaultPlanStage()
        self.execute_stage = execute_stage or DefaultExecuteStage()
        self.verify_stage = verify_stage or DefaultVerifyStage()
        self.iterate_stage = iterate_stage or DefaultIterateStage()
        self.stop_conditions = stop_conditions or default_stop_conditions()
        self._context: LoopContext | None = None

    @property
    def context(self) -> LoopContext:
        if self._context is None:
            raise LoopStageError("LoopController has not been initialized with a context.")
        return self._context

    @property
    def state(self) -> LoopState:
        return self.context.state

    @property
    def is_cancelled(self) -> bool:
        return self._context is not None and self.context.state.cancel_requested

    def request_cancel(self) -> None:
        """Request cancellation. Honoured at the next stage boundary."""
        if self._context is not None:
            self.context.state = self.context.state.request_cancel()

    def initialize(self, context: LoopContext) -> LoopContext:
        """Bind the controller to a context and mark the loop as started."""
        self._context = context
        context.update_state(status=LoopStatus.CREATED, started_at=utc_now())
        return context

    async def run(self, context: LoopContext) -> LoopResult:
        """Run the full loop to completion and return a :class:`LoopResult`."""
        self.initialize(context)
        await self._publish(
            EventType.LOOP_STARTED,
            payload={"goal": context.state.goal, "max_iterations": context.state.max_iterations},
            message="Starting work on the objective.",
        )
        try:
            return await self._run_loop()
        except LoopCancelledError as exc:
            return await self._finalize(LoopFinalStatus.CANCELLED, failure_reason=str(exc))
        except Exception as exc:  # noqa: BLE001 - loop must never crash unbounded
            logger.bind(
                event="loop.crash", loop_id=context.loop_id, error=type(exc).__name__
            ).exception("Loop terminated with an unexpected error: {}", str(exc))
            await self._publish(
                EventType.LOOP_FAILED,
                payload={"error": str(exc)},
                message="The loop failed unexpectedly.",
            )
            return await self._finalize(LoopFinalStatus.FAILED, failure_reason=str(exc))

    async def _run_loop(self) -> LoopResult:
        # DISCOVER runs once.
        await self._run_stage(self.discover_stage, StageStatus.DISCOVER)

        return await self._iteration_loop()

    async def _iteration_loop(self) -> LoopResult:
        """Main iteration loop: PLAN → EXECUTE → VERIFY → DECIDE → ITERATE."""
        context = self.context
        while True:
            # Begin a new iteration (1-indexed). The hard iteration limit is
            # enforced by MaxIterationsCondition in DECIDE, evaluated *after*
            # each attempt, so the loop never exceeds max_iterations attempts.
            context.state = context.state.begin_iteration()
            await self._publish(
                EventType.LOOP_ITERATION_STARTED,
                payload={"iteration": context.state.iteration_count},
                message=f"Starting iteration {context.state.iteration_count}.",
            )

            self._check_cancelled()
            # PLAN
            await self._run_stage(self.plan_stage, StageStatus.PLAN)

            # If there is nothing to execute, we cannot make progress.
            if (
                context.state.next_action is None
                or context.state.next_action.action_type.value == "none"
            ):
                return await self._finalize(
                    LoopFinalStatus.FAILED,
                    failure_reason="No actionable plan could be produced.",
                )

            self._check_cancelled()
            # EXECUTE
            await self._run_stage(self.execute_stage, StageStatus.EXECUTE)

            # If a tool requires confirmation, pause the loop here (before
            # VERIFY). The loop is resumed via resume_after_confirmation.
            if context.state.confirmation_required:
                return await self._pause_for_confirmation()

            self._check_cancelled()
            # VERIFY
            await self._verify_and_decide()

            decision = self._evaluate_stop_conditions()
            await self._publish(
                EventType.LOOP_STAGE_COMPLETED,
                payload={
                    "stage": StageStatus.DECIDE.value,
                    "decision": decision.status.value if decision.status else None,
                },
                message=_STAGE_MESSAGES[StageStatus.DECIDE],
            )

            if decision.should_stop:
                return await self._finalize(
                    decision.status or LoopFinalStatus.FAILED,
                    failure_reason=(
                        decision.reason if decision.status is not LoopFinalStatus.SUCCESS else None
                    ),
                    stopped_reason=decision.reason,
                )

            # ITERATE
            await self._run_stage(self.iterate_stage, StageStatus.ITERATE)

            await self._publish(
                EventType.LOOP_ITERATION_COMPLETED,
                payload={"iteration": context.state.iteration_count},
                message=f"Iteration {context.state.iteration_count} complete.",
            )

    async def _verify_and_decide(self) -> None:
        """Run VERIFY, publish verification outcome, and record the iteration."""
        context = self.context
        await self._run_stage(self.verify_stage, StageStatus.VERIFY)

        verification = context.state.last_verification
        if verification is not None and verification.passed:
            await self._publish(
                EventType.LOOP_VERIFICATION_PASSED,
                payload={
                    "iteration": context.state.iteration_count,
                    "summary": verification.summary,
                },
                message="Verification passed.",
            )
        elif verification is not None and not verification.passed:
            await self._publish(
                EventType.LOOP_VERIFICATION_FAILED,
                payload={
                    "summary": verification.summary,
                    "iteration": context.state.iteration_count,
                },
                message=_STAGE_MESSAGES[StageStatus.ITERATE],
            )
        self._record_iteration()

    async def _pause_for_confirmation(self) -> LoopResult:
        """Pause the loop pending user confirmation of a tool call."""
        context = self.context
        confirm = context.state.confirmation_request or {}
        confirmation_id = confirm.get("confirmation_id")
        tool_name = confirm.get("tool_name", "unknown")
        context.update_state(status=LoopStatus.WAITING_FOR_CONFIRMATION)
        await self._publish(
            EventType.LOOP_WAITING_FOR_CONFIRMATION,
            payload={
                "confirmation_id": confirmation_id,
                "tool": tool_name,
                "iteration": context.state.iteration_count,
            },
            message=(
                f"Tool '{tool_name}' requires confirmation. "
                f"The loop is paused until it is approved or rejected."
            ),
        )
        return await self._finalize(
            LoopFinalStatus.WAITING_FOR_CONFIRMATION,
            failure_reason=None,
            stopped_reason=f"Waiting for confirmation of tool '{tool_name}'.",
        )

    async def resume_after_confirmation(
        self, context: LoopContext, confirmation_id: str
    ) -> LoopResult:
        """Resume a paused loop after a confirmation was approved.

        Executes the confirmed tool, then continues from VERIFY → DECIDE →
        (ITERATE or done). The same task_id, loop_id, session_id, and
        iteration continue — no second task is created.
        """
        self._context = context
        await self._publish(
            EventType.LOOP_RESUMED,
            payload={
                "confirmation_id": confirmation_id,
                "iteration": context.state.iteration_count,
            },
            message="Loop resumed after confirmation approval.",
        )
        # EXECUTE the confirmed tool.
        await self._run_stage_executed_confirmed(confirmation_id)

        # If the confirmed execution itself requires confirmation (rare),
        # pause again.
        if context.state.confirmation_required:
            return await self._pause_for_confirmation()

        self._check_cancelled()
        # VERIFY → DECIDE
        await self._verify_and_decide()

        decision = self._evaluate_stop_conditions()
        await self._publish(
            EventType.LOOP_STAGE_COMPLETED,
            payload={
                "stage": StageStatus.DECIDE.value,
                "decision": decision.status.value if decision.status else None,
            },
            message=_STAGE_MESSAGES[StageStatus.DECIDE],
        )
        if decision.should_stop:
            return await self._finalize(
                decision.status or LoopFinalStatus.FAILED,
                failure_reason=(
                    decision.reason if decision.status is not LoopFinalStatus.SUCCESS else None
                ),
                stopped_reason=decision.reason,
            )

        # ITERATE then continue the normal loop.
        await self._run_stage(self.iterate_stage, StageStatus.ITERATE)
        await self._publish(
            EventType.LOOP_ITERATION_COMPLETED,
            payload={"iteration": context.state.iteration_count},
            message=f"Iteration {context.state.iteration_count} complete.",
        )
        return await self._iteration_loop()

    async def resume_after_rejection(
        self, context: LoopContext, confirmation_id: str, *, reason: str
    ) -> LoopResult:
        """Resume a paused loop after a confirmation was rejected.

        The rejected tool is NOT executed. An observation of the rejection is
        recorded, verification fails, and the loop iterates (or fails safely).
        """
        self._context = context
        await self._publish(
            EventType.LOOP_RESUMED,
            payload={
                "confirmation_id": confirmation_id,
                "rejected": True,
                "iteration": context.state.iteration_count,
            },
            message="Loop resumed after confirmation rejection.",
        )
        # Record the rejection as the execution result (no tool runs).
        exec_result = self.execute_stage.rejected_result(context, reason)
        context.update_state(
            current_stage=StageStatus.EXECUTE,
            status=LoopStatus.EXECUTING,
            last_execution=exec_result,
            execution_results=[*context.state.execution_results, exec_result],
            confirmation_required=False,
            confirmation_request=None,
            failed_steps=[*context.state.failed_steps, exec_result.execution_id],
            last_error=exec_result.error,
        )

        self._check_cancelled()
        # VERIFY → DECIDE (verification will fail since execution was rejected).
        await self._verify_and_decide()

        decision = self._evaluate_stop_conditions()
        await self._publish(
            EventType.LOOP_STAGE_COMPLETED,
            payload={
                "stage": StageStatus.DECIDE.value,
                "decision": decision.status.value if decision.status else None,
            },
            message=_STAGE_MESSAGES[StageStatus.DECIDE],
        )
        if decision.should_stop:
            return await self._finalize(
                decision.status or LoopFinalStatus.FAILED,
                failure_reason=(
                    decision.reason if decision.status is not LoopFinalStatus.SUCCESS else None
                ),
                stopped_reason=decision.reason,
            )

        # ITERATE then continue the normal loop.
        await self._run_stage(self.iterate_stage, StageStatus.ITERATE)
        await self._publish(
            EventType.LOOP_ITERATION_COMPLETED,
            payload={"iteration": context.state.iteration_count},
            message=f"Iteration {context.state.iteration_count} complete.",
        )
        return await self._iteration_loop()

    async def _run_stage_executed_confirmed(self, confirmation_id: str) -> None:
        """Execute a confirmed tool via the execute stage and store the result."""
        context = self.context
        await self._publish(
            EventType.LOOP_STAGE_STARTED,
            payload={
                "stage": StageStatus.EXECUTE.value,
                "iteration": context.state.iteration_count,
                "confirmed": True,
            },
            message="Executing confirmed tool.",
        )
        exec_result = await self.execute_stage.execute_confirmed(context, confirmation_id)
        context.update_state(
            current_stage=StageStatus.EXECUTE,
            status=LoopStatus.EXECUTING,
            last_execution=exec_result,
            execution_results=[*context.state.execution_results, exec_result],
        )
        if exec_result.status == "success":
            context.update_state(
                completed_steps=[*context.state.completed_steps, exec_result.execution_id]
            )
        else:
            context.update_state(
                failed_steps=[*context.state.failed_steps, exec_result.execution_id],
                last_error=exec_result.error,
            )
        await self._publish(
            EventType.LOOP_STAGE_COMPLETED,
            payload={
                "stage": StageStatus.EXECUTE.value,
                "iteration": context.state.iteration_count,
            },
            message="Stage execute complete.",
        )

    async def _run_stage(self, stage: Any, stage_status: StageStatus) -> None:
        """Run a single stage, wrapping it with stage-started/completed events."""
        context = self.context
        await self._publish(
            EventType.LOOP_STAGE_STARTED,
            payload={"stage": stage_status.value, "iteration": context.state.iteration_count},
            message=_STAGE_MESSAGES.get(stage_status, f"Running stage {stage_status.value}."),
        )
        try:
            await stage.run(context)
        except LoopCancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - isolate stage failures
            raise LoopStageError(f"Stage '{stage.name}' failed: {exc}") from exc
        await self._publish(
            EventType.LOOP_STAGE_COMPLETED,
            payload={"stage": stage_status.value, "iteration": context.state.iteration_count},
            message=f"Stage {stage_status.value} complete.",
        )

    def _evaluate_stop_conditions(self) -> StopDecision:
        for condition in self.stop_conditions:
            decision = condition.evaluate(self.context)
            if decision.should_stop:
                return decision
        return StopDecision.continue_loop()

    def _check_cancelled(self) -> None:
        if self.context.state.cancel_requested:
            raise LoopCancelledError("Loop cancelled by request.")

    def _record_iteration(self) -> None:
        """Append an :class:`IterationRecord` for the current attempt.

        Captures the executed action, execution result, verification outcome,
        accumulated changes, and errors. Called once per completed attempt
        (after VERIFY), so the final stopping attempt is recorded too.
        """
        state = self.context.state
        if any(r.iteration_number == state.iteration_count for r in state.iteration_history):
            return
        record = IterationRecord(
            iteration_number=state.iteration_count,
            stage=StageStatus.ITERATE.value,
            action=state.next_action,
            result=state.last_execution,
            verification=state.last_verification,
            changes=list(state.changes),
            errors=([state.last_error] if state.last_error else []),
        )
        self.context.state = state.add_iteration_record(record)

    async def _finalize(
        self,
        status: LoopFinalStatus,
        *,
        failure_reason: str | None = None,
        stopped_reason: str | None = None,
    ) -> LoopResult:
        context = self.context
        state = context.state
        final_response = None
        if status is LoopFinalStatus.SUCCESS:
            final_response = state.last_execution.output if state.last_execution else None
            state = state.evolve(status=LoopStatus.COMPLETED, current_stage=StageStatus.DONE)
        elif status is LoopFinalStatus.CANCELLED:
            state = state.evolve(status=LoopStatus.CANCELLED, current_stage=StageStatus.DONE)
        elif status is LoopFinalStatus.MAX_ITERATIONS_REACHED:
            state = state.evolve(
                status=LoopStatus.MAX_ITERATIONS_REACHED, current_stage=StageStatus.DONE
            )
        elif status is LoopFinalStatus.WAITING_FOR_CONFIRMATION:
            # Non-terminal pause: the loop is suspended, not done. The state
            # is already WAITING_FOR_CONFIRMATION (set by _pause_for_confirmation).
            pass
        else:
            state = state.evolve(status=LoopStatus.FAILED, current_stage=StageStatus.DONE)
        context.state = state

        result = LoopResult.from_state(
            state,
            final_status=status,
            final_response=final_response,
            failure_reason=failure_reason,
            stopped_reason=stopped_reason,
        )

        # Publish the appropriate terminal event.
        if status is LoopFinalStatus.SUCCESS:
            await self._publish(
                EventType.LOOP_COMPLETED,
                payload={"iterations": result.iterations_used},
                message="Task completed successfully.",
            )
        elif status is LoopFinalStatus.CANCELLED:
            await self._publish(
                EventType.LOOP_STOPPED,
                payload={"reason": stopped_reason},
                message="Task cancelled.",
            )
        elif status is LoopFinalStatus.MAX_ITERATIONS_REACHED:
            await self._publish(
                EventType.LOOP_STOPPED,
                payload={"iterations": result.iterations_used, "reason": stopped_reason},
                message="Reached the iteration limit without completing the task.",
            )
        elif status is LoopFinalStatus.WAITING_FOR_CONFIRMATION:
            # No terminal event — the loop is paused, not terminated.
            # LOOP_WAITING_FOR_CONFIRMATION was already published by _pause_for_confirmation.
            pass
        else:
            await self._publish(
                EventType.LOOP_FAILED,
                payload={"reason": failure_reason},
                message="The task could not be completed.",
            )
        return result

    async def _publish(
        self, event_type: EventType, *, payload: dict[str, Any] | None, message: str
    ) -> None:
        context = self.context
        meta = {
            "loop_id": context.loop_id,
            "task_id": context.task_id,
            "session_id": context.session_id,
            "iteration_number": context.state.iteration_count,
            "message": message,
        }
        await context.event_bus.publish(
            Event.create(
                event_type,
                task_id=context.task_id,
                session_id=context.session_id,
                payload={**(payload or {}), "message": message, "loop_id": context.loop_id},
                metadata=meta,
            )
        )

    @staticmethod
    def build_state(
        task: Task,
        *,
        goal: str,
        success_criteria: list[str] | None = None,
        max_iterations: int = 5,
        session_id: str | None = None,
    ) -> LoopState:
        """Helper to construct an initial :class:`LoopState` from a task."""
        return LoopState(
            task_id=task.task_id,
            session_id=session_id or task.session_id,
            goal=goal,
            success_criteria=list(success_criteria or []),
            max_iterations=max_iterations,
        )
