"""Typed event definitions for the event bus.

Events carry enough metadata for future voice/status reporting:
* event_id, timestamp, event_type
* task_id, agent_id, session_id (when applicable)
* payload, metadata
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.core.identifiers import generate_id, utc_now

__all__ = ["EventType", "BaseEvent", "Event"]


class EventType(StrEnum):
    """All event types produced and consumed by the runtime."""

    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"

    AGENT_STARTED = "agent.started"
    AGENT_COMPLETED = "agent.completed"
    AGENT_FAILED = "agent.failed"

    TOOL_STARTED = "tool.started"
    TOOL_COMPLETED = "tool.completed"

    MEMORY_UPDATED = "memory.updated"

    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_COMPLETED = "workflow.completed"

    VOICE_STATUS = "voice.status"

    # --- Universal Loop Engine events (Phase 1) ---
    LOOP_STARTED = "loop.started"
    LOOP_STAGE_STARTED = "loop.stage.started"
    LOOP_STAGE_COMPLETED = "loop.stage.completed"
    LOOP_ITERATION_STARTED = "loop.iteration.started"
    LOOP_ITERATION_COMPLETED = "loop.iteration.completed"
    LOOP_VERIFICATION_PASSED = "loop.verification.passed"
    LOOP_VERIFICATION_FAILED = "loop.verification.failed"
    LOOP_COMPLETED = "loop.completed"
    LOOP_STOPPED = "loop.stopped"
    LOOP_FAILED = "loop.failed"

    # --- Phase 4: confirmation lifecycle events ---
    LOOP_WAITING_FOR_CONFIRMATION = "loop.waiting_for_confirmation"
    LOOP_RESUMED = "loop.resumed"

    # --- LLM events (Phase 2) ---
    # No prompt contents, completions, API keys, or authorization headers are
    # ever placed in these events — only provider/model/latency/success info.
    LLM_REQUEST_STARTED = "llm.request.started"
    LLM_REQUEST_COMPLETED = "llm.request.completed"
    LLM_REQUEST_FAILED = "llm.request.failed"
    MODEL_SELECTED = "model.selected"
    MODEL_FALLBACK = "model.fallback"
    MODEL_UNAVAILABLE = "model.unavailable"

    # --- Tool execution events (Phase 3) ---
    TOOL_CALL_REQUESTED = "tool.call.requested"
    # --- Tool ecosystem events (Phase 8.1) ---
    # Discovery payloads carry only tool identity/category/risk + correlation
    # ids — never arguments, prompts, output, or secrets.
    TOOL_DISCOVERED = "tool.discovered"
    TOOL_CALL_STARTED = "tool.call.started"
    TOOL_CALL_COMPLETED = "tool.call.completed"
    TOOL_CALL_FAILED = "tool.call.failed"
    TOOL_POLICY_DENIED = "tool.policy.denied"
    TOOL_CONFIRMATION_REQUIRED = "tool.confirmation.required"
    TOOL_CONFIRMATION_APPROVED = "tool.confirmation.approved"
    TOOL_CONFIRMATION_REJECTED = "tool.confirmation.rejected"
    TOOL_CONFIRMATION_EXPIRED = "tool.confirmation.expired"
    OBSERVATION_CREATED = "observation.created"
    LLM_FALLBACK = "llm.fallback"

    # --- Memory events (Phase 6) ---
    # No memory content is ever placed in these events — only memory_id,
    # memory_type, source, scope, scores, and counts.
    # MEMORY_UPDATED already exists above (Phase 0).
    MEMORY_CREATED = "memory.created"
    MEMORY_DELETED = "memory.deleted"
    MEMORY_RETRIEVED = "memory.retrieved"
    MEMORY_SEARCH_FAILED = "memory.search.failed"
    MEMORY_WRITE_REJECTED = "memory.write.rejected"
    MEMORY_CONSOLIDATED = "memory.consolidated"

    # --- Agent orchestration events (Phase 7) ---
    # No secrets, instructions, or memory content are ever placed in these
    # events — only agent_id, version, lifecycle, routing metadata, and
    # objective outcome facts (counts/status).
    AGENT_SELECTED = "agent.selected"
    AGENT_DISPATCH_STARTED = "agent.dispatch.started"
    AGENT_DISPATCH_COMPLETED = "agent.dispatch.completed"
    AGENT_DISPATCH_FAILED = "agent.dispatch.failed"
    AGENT_CREATED = "agent.created"
    AGENT_UPDATED = "agent.updated"
    AGENT_DEPRECATED = "agent.deprecated"
    AGENT_RETIRED = "agent.retired"
    AGENT_EVALUATION_STARTED = "agent.evaluation.started"
    AGENT_EVALUATION_COMPLETED = "agent.evaluation.completed"
    AGENT_VERSION_CREATED = "agent.version.created"


class BaseEvent(BaseModel):
    """Base model for all events.

    Subclasses (or instances of :class:`Event`) carry a typed ``event_type``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str = Field(default_factory=lambda: generate_id("evt"))
    event_type: EventType
    timestamp: datetime = Field(default_factory=utc_now)
    task_id: str | None = None
    agent_id: str | None = None
    session_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def create(
        cls,
        event_type: EventType,
        *,
        payload: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        task_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> Event:
        return Event(
            event_type=event_type,
            payload=payload or {},
            metadata=metadata or {},
            task_id=task_id,
            agent_id=agent_id,
            session_id=session_id,
        )


class Event(BaseEvent):
    """Concrete, general-purpose event used by the runtime."""

    pass
