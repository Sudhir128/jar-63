"""Stop conditions: the controlled termination rules for a loop.

A :class:`StopCondition` is evaluated after verification and before another
iteration. If any condition fires, the loop terminates with a clear reason.
Conditions are pluggable — future phases may register new ones without
touching the controller.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

from app.runtime.loop.loop_context import LoopContext
from app.runtime.loop.loop_result import LoopFinalStatus

__all__ = [
    "StopDecision",
    "StopCondition",
    "MaxIterationsCondition",
    "SuccessCondition",
    "CancellationCondition",
    "FailureCondition",
    "TimeoutCondition",
]


@dataclass(frozen=True)
class StopDecision:
    """The verdict of a stop condition.

    ``should_stop`` is True when the loop must terminate now; ``status`` is
    the terminal status to assign, and ``reason`` is a human-readable message
    suitable for events and the final result.
    """

    should_stop: bool
    status: LoopFinalStatus | None = None
    reason: str = ""

    @classmethod
    def continue_loop(cls) -> StopDecision:
        return cls(should_stop=False)


class StopCondition(abc.ABC):
    """Abstract stop condition evaluated after verification."""

    @abc.abstractmethod
    def evaluate(self, context: LoopContext) -> StopDecision:
        """Return a :class:`StopDecision` for the current state."""


class MaxIterationsCondition(StopCondition):
    """Stop when the iteration count reaches the configured hard limit."""

    def evaluate(self, context: LoopContext) -> StopDecision:
        state = context.state
        if state.iteration_count >= state.max_iterations:
            return StopDecision(
                should_stop=True,
                status=LoopFinalStatus.MAX_ITERATIONS_REACHED,
                reason=(
                    f"Reached max iterations ({state.max_iterations}) "
                    f"without meeting success criteria."
                ),
            )
        return StopDecision.continue_loop()


class SuccessCondition(StopCondition):
    """Stop with SUCCESS when verification passed and evidence exists.

    A loop may only declare success when verification produced objective
    evidence. Execution succeeding alone is insufficient.
    """

    def evaluate(self, context: LoopContext) -> StopDecision:
        verification = context.state.last_verification
        if verification is not None and verification.passed and verification.has_evidence:
            return StopDecision(
                should_stop=True,
                status=LoopFinalStatus.SUCCESS,
                reason="Verification passed with objective evidence.",
            )
        return StopDecision.continue_loop()


class CancellationCondition(StopCondition):
    """Stop with CANCELLED when cancellation was requested."""

    def evaluate(self, context: LoopContext) -> StopDecision:
        if context.state.cancel_requested:
            return StopDecision(
                should_stop=True,
                status=LoopFinalStatus.CANCELLED,
                reason="Loop cancelled by request.",
            )
        return StopDecision.continue_loop()


class FailureCondition(StopCondition):
    """Stop with FAILED when a terminal failure occurred.

    Fires when the last execution failed and retries are not allowed, or when
    a stage raised an unrecoverable error recorded in ``last_error`` while
    verification is impossible.
    """

    def __init__(self, *, unrecoverable_only: bool = True) -> None:
        self._unrecoverable_only = unrecoverable_only

    def evaluate(self, context: LoopContext) -> StopDecision:
        state = context.state
        last_exec = state.last_execution
        if (
            last_exec is not None
            and last_exec.status == "failed"
            and (not context.policy.allow_retry or not self._allow_retry_for(state))
        ):
            return StopDecision(
                should_stop=True,
                status=LoopFinalStatus.FAILED,
                reason=f"Execution failed: {last_exec.error or 'unknown error'}",
            )
        return StopDecision.continue_loop()

    @staticmethod
    def _allow_retry_for(state: object) -> bool:
        # Without per-step retry counters yet, allow retry unless explicitly
        # disabled. The hard iteration limit still bounds total retries.
        return True


class TimeoutCondition(StopCondition):
    """Stop when the loop exceeds its configured execution time budget."""

    def evaluate(self, context: LoopContext) -> StopDecision:
        state = context.state
        if state.started_at is None:
            return StopDecision.continue_loop()
        from app.core.identifiers import utc_now

        elapsed = (utc_now() - state.started_at).total_seconds()
        if elapsed >= context.policy.max_execution_time_seconds:
            return StopDecision(
                should_stop=True,
                status=LoopFinalStatus.FAILED,
                reason=(
                    f"Execution timed out after {elapsed:.1f}s "
                    f"(limit {context.policy.max_execution_time_seconds}s)."
                ),
            )
        return StopDecision.continue_loop()


def default_stop_conditions() -> list[StopCondition]:
    """The default, ordered set of stop conditions for a loop.

    Order matters: cancellation and success are checked before the hard
    iteration limit so a just-succeeded iteration is not misreported.
    """
    return [
        CancellationCondition(),
        SuccessCondition(),
        MaxIterationsCondition(),
        TimeoutCondition(),
        FailureCondition(),
    ]
