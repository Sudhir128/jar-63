"""Loop result and final status.

A :class:`LoopResult` is the immutable, observable outcome of a completed
(terminated) loop. Every loop terminates with exactly one
:class:`LoopFinalStatus`; there is no such thing as a loop that runs forever.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.runtime.loop.verification.verification_result import VerificationEvidence

__all__ = ["LoopFinalStatus", "LoopResult"]


class LoopFinalStatus(StrEnum):
    """Terminal statuses a loop may reach.

    A loop is only ``SUCCESS`` when verification produced evidence that the
    objective was achieved. Execution succeeding is not sufficient.

    ``WAITING_FOR_CONFIRMATION`` is a non-terminal pause point: the loop has
    suspended execution pending user approval of a high-risk tool call. It is
    resumed via :meth:`LoopController.resume_after_confirmation`.
    """

    SUCCESS = "success"
    INCOMPLETE = "incomplete"
    FAILED = "failed"
    CANCELLED = "cancelled"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"
    WAITING_FOR_CONFIRMATION = "waiting_for_confirmation"


class LoopResult(BaseModel):
    """The final, immutable outcome of a loop run."""

    model_config = ConfigDict(frozen=True)

    loop_id: str
    task_id: str
    session_id: str | None = None
    final_status: LoopFinalStatus
    final_response: Any = None
    iterations_used: int = 0
    success: bool = False
    verification_evidence: list[VerificationEvidence] = Field(default_factory=list)
    completed_work: list[str] = Field(default_factory=list)
    remaining_work: list[str] = Field(default_factory=list)
    failure_reason: str | None = None
    stopped_reason: str | None = None

    @classmethod
    def from_state(cls, state: Any, *, final_status: LoopFinalStatus, **kwargs: Any) -> LoopResult:
        """Build a result from a :class:`LoopState` snapshot.

        Accepts ``state`` loosely typed to avoid an import cycle with
        :mod:`app.runtime.loop.loop_state`.
        """
        success = final_status is LoopFinalStatus.SUCCESS
        return cls(
            loop_id=state.loop_id,
            task_id=state.task_id,
            session_id=state.session_id,
            final_status=final_status,
            iterations_used=state.iteration_count,
            success=success,
            verification_evidence=[
                ev for vr in (state.verification_results or []) for ev in vr.evidence
            ],
            completed_work=list(state.completed_steps or []),
            remaining_work=list(state.remaining_work or []),
            **kwargs,
        )
