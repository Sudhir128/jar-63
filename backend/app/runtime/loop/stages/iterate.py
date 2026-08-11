"""Iterate stage.

When verification fails, the iterate stage records the failure, identifies the
weakest failed criterion, updates state, determines a meaningful next action,
and increments the iteration count.

The loop never blindly repeats the same failed action. The iteration record
captures what changed so history is preserved for memory, debugging, and
observability.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.runtime.loop.loop_context import LoopContext
from app.runtime.loop.loop_state import (
    ActionType,
    LoopStatus,
    NextAction,
    StageStatus,
)
from app.runtime.loop.stages.base import LoopStage

logger = get_logger("loop.iterate")

__all__ = ["IterationResult", "IterateStage", "DefaultIterateStage"]


class IterationResult:
    """Marker type for the iterate stage's outcome (state is the source of truth)."""

    pass


class IterateStage(LoopStage):
    """Abstract iterate stage contract."""

    name = "iterate"

    async def run(self, context: LoopContext) -> LoopContext:
        result_action = await self.iterate(context)
        context.update_state(
            current_stage=StageStatus.ITERATE,
            status=LoopStatus.ITERATING,
            next_action=result_action,
        )
        return context

    async def iterate(self, context: LoopContext) -> NextAction:
        """Determine the next action after a failed verification."""


class DefaultIterateStage(IterateStage):
    """Deterministic iterate stage.

    On verification failure it records the weakest failed check as a blocker,
    notes the change ("adjusting plan after verification failure"), and
    requests a ``FIX`` action targeting the same agent so the next iteration
    attempts a corrected execution. The hard iteration limit still bounds the
    total number of retries.
    """

    async def iterate(self, context: LoopContext) -> NextAction:
        verification = context.state.last_verification
        weakest = _weakest_failed(verification)
        change = f"Iteration {context.state.iteration_count}: verification failed"
        if weakest is not None:
            change += f" (weakest check: {weakest.check})"
        blockers = [*context.state.blockers]
        if weakest is not None:
            blockers.append(f"Failed check: {weakest.check}")

        context.update_state(
            changes=[*context.state.changes, change],
            blockers=blockers,
        )

        action = context.state.next_action
        target = action.target if action else None
        return NextAction(
            action_type=ActionType.FIX,
            target=target,
            description=f"Fix failing verification: {weakest.check if weakest else 'objective not met'}.",
            reasoning="Verification failed; adjusting execution to address the weakest failed check.",
        )


def _weakest_failed(verification: object | None):
    """Return the first failed evidence item, or None."""
    if verification is None:
        return None
    evidence = getattr(verification, "evidence", None) or []
    for ev in evidence:
        if not getattr(ev, "passed", True):
            return ev
    return None
