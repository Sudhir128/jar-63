"""Task and session domain models for the runtime foundation."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.core.identifiers import generate_id, utc_now

__all__ = ["TaskStatus", "Task", "SessionStatus", "Session"]


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_FOR_CONFIRMATION = "waiting_for_confirmation"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Task(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str = Field(default_factory=lambda: generate_id("task"))
    session_id: str | None = None
    agent_id: str | None = None
    input: Any = None
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str | None = None
    created_at: Any = Field(default_factory=utc_now)
    updated_at: Any = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionStatus(StrEnum):
    ACTIVE = "active"
    CLOSED = "closed"


class Session(BaseModel):
    model_config = ConfigDict(extra="allow")

    session_id: str = Field(default_factory=lambda: generate_id("session"))
    user_id: str | None = None
    status: SessionStatus = SessionStatus.ACTIVE
    created_at: Any = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)
