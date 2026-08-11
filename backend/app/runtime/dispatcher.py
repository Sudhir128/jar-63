"""Dispatcher: routes tasks to the appropriate agent.

The full routing/policy logic is deferred to later phases. This module
defines the contract and a minimal, honest default that delegates execution
to a single explicitly-selected agent via the registry.
"""

from __future__ import annotations

import abc

from app.agents.interface import AgentContext, AgentInterface, AgentResult
from app.agents.registry import AgentRegistry
from app.core.exceptions import AgentNotFoundError
from app.core.logging import get_logger
from app.events import Event, EventBus, EventType
from app.runtime.models import Task

logger = get_logger("runtime.dispatcher")

__all__ = ["Dispatcher", "DefaultDispatcher"]


class Dispatcher(abc.ABC):
    """Abstract contract for routing tasks to agents."""

    @abc.abstractmethod
    async def dispatch(self, task: Task) -> AgentResult:
        """Route ``task`` to an agent and return its result."""


class DefaultDispatcher(Dispatcher):
    """Minimal dispatcher that runs a task against an explicitly-chosen agent.

    It performs no planning, routing, or policy decisions yet — those are
    responsibilities of later phases. When ``task.agent_id`` is set, the
    dispatcher looks the agent up and executes it; otherwise it raises.
    """

    def __init__(self, registry: AgentRegistry, event_bus: EventBus) -> None:
        self._registry = registry
        self._event_bus = event_bus

    async def dispatch(self, task: Task) -> AgentResult:
        agent_id = task.agent_id
        if not agent_id:
            raise AgentNotFoundError("Task has no agent_id; routing is not implemented yet")

        agent: AgentInterface = self._registry.get(agent_id)
        context = AgentContext(
            task_id=task.task_id,
            session_id=task.session_id,
            input=task.input,
            metadata=task.metadata,
        )

        await self._event_bus.publish(
            Event.create(
                EventType.AGENT_STARTED,
                task_id=task.task_id,
                agent_id=agent_id,
                session_id=task.session_id,
                payload={"input_type": type(task.input).__name__},
            )
        )

        await agent.on_start(context)
        try:
            result = await agent.execute(context)
        except Exception as exc:
            await agent.on_error(exc)
            await self._event_bus.publish(
                Event.create(
                    EventType.AGENT_FAILED,
                    task_id=task.task_id,
                    agent_id=agent_id,
                    session_id=task.session_id,
                    payload={"error": str(exc)},
                )
            )
            raise
        await agent.on_complete(result)

        await self._event_bus.publish(
            Event.create(
                EventType.AGENT_COMPLETED,
                task_id=task.task_id,
                agent_id=agent_id,
                session_id=task.session_id,
                payload={"status": result.status.value},
            )
        )
        return result
