"""Observation model.

An :class:`Observation` represents what the system learned after executing an
action (a tool call or agent execution). It is distinct from the execution
result and from verification:

* **Plan**     — what the system intends to do.
* **Action**   — the concrete action taken (execute a tool/agent).
* **Observation** — what was learned from the action (the result, normalized).
* **Verification** — whether the objective was achieved (independent check).

Observations feed back into the loop state so the next PLAN stage (LLM or
deterministic) can reason about what happened.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.core.identifiers import generate_id, utc_now

__all__ = ["ObservationType", "Observation"]


class ObservationType(StrEnum):
    """The kind of thing observed."""

    TOOL_RESULT = "tool_result"
    TOOL_FAILURE = "tool_failure"
    TOOL_DENIED = "tool_denied"
    TOOL_CONFIRMATION_REQUIRED = "tool_confirmation_required"
    AGENT_RESULT = "agent_result"
    AGENT_FAILURE = "agent_failure"
    PLAN = "plan"
    INFO = "info"


class Observation(BaseModel):
    """A typed record of what the loop learned from an action.

    ``content`` is the normalized, safe representation of the result — never
    raw exceptions or secrets. ``tool_call_id`` correlates the observation
    with the tool call that produced it (when applicable).
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    observation_id: str = Field(default_factory=lambda: generate_id("obs"))
    source: str = ""
    type: ObservationType = ObservationType.INFO
    content: Any = None
    success: bool = True
    timestamp: datetime = Field(default_factory=utc_now)
    tool_call_id: str | None = None
    tool_name: str | None = None
    task_id: str | None = None
    session_id: str | None = None
    iteration: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_tool_result(
        cls,
        *,
        tool_name: str,
        tool_call_id: str | None,
        result: Any,
        success: bool,
        task_id: str | None = None,
        session_id: str | None = None,
        iteration: int | None = None,
        error: str | None = None,
    ) -> Observation:
        obs_type = ObservationType.TOOL_RESULT if success else ObservationType.TOOL_FAILURE
        content = error if not success and error else result
        return cls(
            source=f"tool:{tool_name}",
            type=obs_type,
            content=content,
            success=success,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            task_id=task_id,
            session_id=session_id,
            iteration=iteration,
        )

    @classmethod
    def from_denial(
        cls,
        *,
        tool_name: str,
        tool_call_id: str | None,
        reason: str,
        task_id: str | None = None,
        session_id: str | None = None,
        iteration: int | None = None,
    ) -> Observation:
        return cls(
            source=f"tool:{tool_name}",
            type=ObservationType.TOOL_DENIED,
            content=reason,
            success=False,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            task_id=task_id,
            session_id=session_id,
            iteration=iteration,
        )

    @classmethod
    def from_confirmation_required(
        cls,
        *,
        tool_name: str,
        tool_call_id: str | None,
        confirmation_id: str,
        reason: str,
        task_id: str | None = None,
        session_id: str | None = None,
        iteration: int | None = None,
    ) -> Observation:
        return cls(
            source=f"tool:{tool_name}",
            type=ObservationType.TOOL_CONFIRMATION_REQUIRED,
            content=reason,
            success=False,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            task_id=task_id,
            session_id=session_id,
            iteration=iteration,
            metadata={"confirmation_id": confirmation_id},
        )
