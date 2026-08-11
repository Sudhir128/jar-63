"""Strongly typed loop state.

``LoopState`` is the single source of truth for a running loop. It is updated
functionally (``model_copy(update=...)``) so that prior snapshots remain valid
for the iteration history. No unstructured ``dict`` is used as the primary
representation.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.core.identifiers import generate_id, utc_now
from app.runtime.loop.verification.verification_result import VerificationResult

__all__ = [
    "LoopStatus",
    "StageStatus",
    "NextAction",
    "ActionType",
    "PlanStep",
    "ExecutionResult",
    "IterationRecord",
    "LoopState",
]


class LoopStatus(StrEnum):
    """Lifecycle status of a loop instance."""

    CREATED = "created"
    DISCOVERING = "discovering"
    PLANNING = "planning"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    ITERATING = "iterating"
    WAITING_FOR_CONFIRMATION = "waiting_for_confirmation"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"


class StageStatus(StrEnum):
    """Identifier for the current stage within the loop."""

    DISCOVER = "discover"
    PLAN = "plan"
    EXECUTE = "execute"
    VERIFY = "verify"
    ITERATE = "iterate"
    DECIDE = "decide"
    DONE = "done"


class ActionType(StrEnum):
    """Kinds of actions a plan may request."""

    EXECUTE_AGENT = "execute_agent"
    EXECUTE_TOOL = "execute_tool"
    COMPLETE = "complete"
    RETRY = "retry"
    FIX = "fix"
    NONE = "none"


class NextAction(BaseModel):
    """The next concrete action the loop should take."""

    model_config = ConfigDict(frozen=True, extra="allow")

    action_type: ActionType = ActionType.NONE
    target: str | None = None
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    reasoning: str = ""


class PlanStep(BaseModel):
    """A single planned step."""

    model_config = ConfigDict(frozen=True, extra="allow")

    step_id: str = Field(default_factory=lambda: generate_id("step"))
    description: str
    action_type: ActionType = ActionType.EXECUTE_AGENT
    target: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    status: str = "pending"


class ExecutionResult(BaseModel):
    """The outcome of executing a single action."""

    model_config = ConfigDict(frozen=True, extra="allow")

    execution_id: str = Field(default_factory=lambda: generate_id("exec"))
    agent_id: str | None = None
    tool_name: str | None = None
    status: str = "success"
    output: Any = None
    error: str | None = None
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None
    duration_ms: int | None = None


class IterationRecord(BaseModel):
    """An immutable record of one loop iteration.

    Preserved in ``LoopState.iteration_history`` for memory, UI, debugging,
    observability, and analytics in later phases.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    iteration_number: int
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None
    stage: str = ""
    action: NextAction | None = None
    result: ExecutionResult | None = None
    verification: VerificationResult | None = None
    changes: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class LoopState(BaseModel):
    """Strongly typed, mutable loop state.

    Updated functionally via :meth:`evolve` so previous snapshots are not
    mutated. Iteration history is append-only.
    """

    model_config = ConfigDict(extra="forbid")

    loop_id: str = Field(default_factory=lambda: generate_id("loop"))
    task_id: str
    session_id: str | None = None
    goal: str
    success_criteria: list[str] = Field(default_factory=list)
    current_stage: StageStatus = StageStatus.DONE
    iteration_count: int = 0
    max_iterations: int = 5
    status: LoopStatus = LoopStatus.CREATED
    cancel_requested: bool = False

    completed_steps: list[str] = Field(default_factory=list)
    failed_steps: list[str] = Field(default_factory=list)
    changes: list[str] = Field(default_factory=list)
    remaining_work: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)

    next_action: NextAction | None = None
    last_execution: ExecutionResult | None = None
    last_verification: VerificationResult | None = None
    execution_results: list[ExecutionResult] = Field(default_factory=list)
    verification_results: list[VerificationResult] = Field(default_factory=list)
    last_error: str | None = None

    # --- Phase 3: LLM-powered loop + tool execution ---
    current_plan: dict[str, Any] | None = None
    current_step: PlanStep | None = None
    pending_tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    observations: list[dict[str, Any]] = Field(default_factory=list)
    tool_call_count: int = 0
    tool_call_count_per_iteration: int = 0
    confirmation_required: bool = False
    confirmation_request: dict[str, Any] | None = None

    iteration_history: list[IterationRecord] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None

    def evolve(self, **updates: Any) -> LoopState:
        """Return a copy of this state with ``updates`` applied.

        Always bumps ``updated_at``. This keeps updates functional and
        observable without mutating the snapshot in place.
        """
        updates.setdefault("updated_at", utc_now())
        return self.model_copy(update=updates)

    def begin_iteration(self) -> LoopState:
        """Advance to the next iteration and record its start."""
        return self.evolve(
            iteration_count=self.iteration_count + 1,
            status=LoopStatus.ITERATING,
        )

    def request_cancel(self) -> LoopState:
        """Mark the loop as cancelled (checked at stage boundaries)."""
        return self.evolve(cancel_requested=True, status=LoopStatus.CANCELLED)

    def add_iteration_record(self, record: IterationRecord) -> LoopState:
        """Append an immutable iteration record (append-only history)."""
        return self.evolve(iteration_history=[*self.iteration_history, record])

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            LoopStatus.COMPLETED,
            LoopStatus.FAILED,
            LoopStatus.CANCELLED,
            LoopStatus.MAX_ITERATIONS_REACHED,
        }
