"""Workflow manager: foundation for multi-step agent workflows.

Full workflow composition (LangGraph integration) is deferred to later
phases. This module defines the contract and a minimal registry of workflow
definitions.
"""

from __future__ import annotations

import abc
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.core.identifiers import generate_id
from app.core.logging import get_logger
from app.events import Event, EventBus, EventType

logger = get_logger("runtime.workflow")

__all__ = ["WorkflowDefinition", "WorkflowStatus", "WorkflowManager"]


class WorkflowStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkflowDefinition(BaseModel):
    """Declarative description of a workflow (steps filled in later phases)."""

    model_config = ConfigDict(frozen=True)

    workflow_id: str = Field(default_factory=lambda: generate_id("wf"))
    name: str
    description: str = ""
    steps: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowManager(abc.ABC):
    """Abstract contract for workflow execution."""

    @abc.abstractmethod
    async def run(self, workflow: WorkflowDefinition) -> Any:
        """Execute a workflow. Concrete execution is implemented in later phases."""


class DefaultWorkflowManager(WorkflowManager):
    """Minimal workflow manager: stores definitions and emits lifecycle events.

    Does not yet execute steps; later phases implement orchestration on top of
    LangGraph without changing this contract.
    """

    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        self._workflows: dict[str, WorkflowDefinition] = {}

    def register(self, workflow: WorkflowDefinition) -> WorkflowDefinition:
        self._workflows[workflow.workflow_id] = workflow
        return workflow

    def list(self) -> list[WorkflowDefinition]:
        return list(self._workflows.values())

    async def run(self, workflow: WorkflowDefinition) -> Any:
        wf_id = workflow.workflow_id
        await self._event_bus.publish(
            Event.create(
                EventType.WORKFLOW_STARTED,
                payload={"workflow_id": wf_id, "name": workflow.name},
            )
        )
        logger.bind(event="workflow.run", workflow_id=wf_id).info(
            "Workflow execution requested for '{}' (orchestration not implemented yet)",
            workflow.name,
        )
        await self._event_bus.publish(
            Event.create(
                EventType.WORKFLOW_COMPLETED,
                payload={"workflow_id": wf_id, "name": workflow.name},
            )
        )
        return {"workflow_id": wf_id, "status": "deferred"}
