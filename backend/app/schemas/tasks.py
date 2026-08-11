"""API schemas for tasks and loops."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "CreateTaskRequest",
    "TaskResponse",
    "LoopIterationResponse",
    "LoopStateResponse",
    "LoopResultResponse",
    "CancelTaskResponse",
    "LoopListResponse",
    "ConfirmationResponse",
    "ConfirmationActionRequest",
    "ResumeTaskResponse",
]


class CreateTaskRequest(BaseModel):
    """Request body for creating and running a loop-backed task."""

    model_config = ConfigDict(extra="forbid")

    goal: str
    agent_id: str | None = None
    input: Any = None
    success_criteria: list[str] = Field(default_factory=list)
    max_iterations: int = Field(default=5, ge=1, le=100)
    expected_output: Any = None
    session_id: str | None = None
    background: bool = False


class TaskResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    session_id: str | None = None
    agent_id: str | None = None
    status: str
    loop_id: str | None = None
    result: Any = None
    error: str | None = None


class LoopIterationResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    iteration_number: int
    stage: str = ""
    action: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    verification: dict[str, Any] | None = None
    changes: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class LoopStateResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    loop_id: str
    task_id: str
    session_id: str | None = None
    goal: str
    status: str
    current_stage: str
    iteration_count: int
    max_iterations: int
    next_action: dict[str, Any] | None = None
    completed_steps: list[str] = Field(default_factory=list)
    failed_steps: list[str] = Field(default_factory=list)
    changes: list[str] = Field(default_factory=list)
    remaining_work: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    last_error: str | None = None
    iterations: list[LoopIterationResponse] = Field(default_factory=list)
    final_result: dict[str, Any] | None = None


class LoopResultResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    loop_id: str
    task_id: str
    final_status: str
    success: bool
    iterations_used: int
    final_response: Any = None
    verification_evidence: list[dict[str, Any]] = Field(default_factory=list)
    completed_work: list[str] = Field(default_factory=list)
    remaining_work: list[str] = Field(default_factory=list)
    failure_reason: str | None = None
    stopped_reason: str | None = None


class CancelTaskResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    cancelled: bool
    message: str


class LoopListResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    loops: list[dict[str, Any]] = Field(default_factory=list)


class ConfirmationResponse(BaseModel):
    """Typed response for a confirmation request (never exposes secrets)."""

    model_config = ConfigDict(frozen=True)

    confirmation_id: str
    task_id: str | None = None
    loop_id: str | None = None
    tool: str
    risk: str
    reason: str = ""
    status: str = "pending"
    iteration: int | None = None


class ConfirmationActionRequest(BaseModel):
    """Request body for approve/reject actions."""

    model_config = ConfigDict(extra="forbid")

    decided_by: str | None = None
    reason: str | None = None


class ResumeTaskResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    loop_id: str
    final_status: str
    success: bool
    iterations_used: int
    final_response: Any = None
    failure_reason: str | None = None
    stopped_reason: str | None = None
