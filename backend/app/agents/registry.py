"""Agent registry: register, unregister, retrieve, list, and check existence."""

from __future__ import annotations

from app.agents.interface import AgentCapability, AgentInfo, AgentInterface
from app.core.exceptions import AgentAlreadyRegisteredError, AgentNotFoundError
from app.core.logging import get_logger

logger = get_logger("agents.registry")

__all__ = ["AgentRegistry"]


class AgentRegistry:
    """In-process registry of agents.

    The registry depends only on :class:`AgentInterface`, never on specific
    agent implementations.
    """

    def __init__(self) -> None:
        self._agents: dict[str, AgentInterface] = {}

    async def register(self, agent: AgentInterface) -> AgentInterface:
        """Register an agent and invoke its ``on_register`` lifecycle hook."""
        agent_id = agent.agent_id
        if agent_id in self._agents:
            raise AgentAlreadyRegisteredError(agent_id)
        await agent.on_register()
        self._agents[agent_id] = agent
        logger.bind(agent_id=agent_id, event="agent.registered").info(
            "Registered agent '{}'", agent.name
        )
        return agent

    async def unregister(self, agent_id: str) -> AgentInterface:
        """Unregister an agent and invoke its ``on_unregister`` hook."""
        agent = self._agents.pop(agent_id, None)
        if agent is None:
            raise AgentNotFoundError(agent_id)
        await agent.on_unregister()
        logger.bind(agent_id=agent_id, event="agent.unregistered").info(
            "Unregistered agent '{}'", agent.name
        )
        return agent

    def get(self, agent_id: str) -> AgentInterface:
        """Return the agent with ``agent_id`` or raise."""
        agent = self._agents.get(agent_id)
        if agent is None:
            raise AgentNotFoundError(agent_id)
        return agent

    def list(self) -> list[AgentInterface]:
        """Return all registered agents."""
        return list(self._agents.values())

    def list_info(self) -> list[AgentInfo]:
        """Return metadata for all registered agents."""
        return [a.info for a in self._agents.values()]

    def exists(self, agent_id: str) -> bool:
        """Return whether an agent with ``agent_id`` is registered."""
        return agent_id in self._agents

    def find_by_capability(self, capability: AgentCapability) -> list[AgentInterface]:
        """Return all agents declaring ``capability``."""
        return [a for a in self._agents.values() if capability in a.capabilities]

    def __len__(self) -> int:
        return len(self._agents)

    def __contains__(self, agent_id: object) -> bool:
        return isinstance(agent_id, str) and agent_id in self._agents
