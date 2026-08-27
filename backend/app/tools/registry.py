"""Tool registry: register, unregister, retrieve, list, and check existence.

Phase 8.1 adds a *discovery* surface: :meth:`list_available` exposes only the
tools available under a supplied policy and decision context. This is a
visibility/filtering mechanism only — the executor still independently
enforces :class:`~app.tools.policy.ToolPolicy` on every execution, so
discovery never replaces authorization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.core.exceptions import ToolAlreadyRegisteredError, ToolNotFoundError
from app.core.logging import get_logger
from app.tools.interface import (
    RiskLevel,
    ToolCategory,
    ToolDecisionContext,
    ToolEnvironment,
    ToolInfo,
    ToolInterface,
)

if TYPE_CHECKING:
    from app.tools.policy import ToolPolicy

logger = get_logger("tools.registry")

__all__ = ["ToolRegistry"]


class ToolRegistry:
    """In-process registry of tools.

    Depends only on :class:`ToolInterface`, never on specific tool
    implementations.
    """

    def __init__(self, *, event_bus: Any = None) -> None:
        self._tools: dict[str, ToolInterface] = {}
        self._event_bus = event_bus

    async def register(self, tool: ToolInterface) -> ToolInterface:
        """Register a tool and invoke its ``on_register`` lifecycle hook."""
        name = tool.name
        if name in self._tools:
            raise ToolAlreadyRegisteredError(name)
        await tool.on_register()
        self._tools[name] = tool
        logger.bind(event="tool.registered", tool=name).info("Registered tool '{}'", name)
        return tool

    async def unregister(self, name: str) -> ToolInterface:
        """Unregister a tool and invoke its ``on_unregister`` hook."""
        tool = self._tools.pop(name, None)
        if tool is None:
            raise ToolNotFoundError(name)
        await tool.on_unregister()
        logger.bind(event="tool.unregistered", tool=name).info("Unregistered tool '{}'", name)
        return tool

    def get(self, name: str) -> ToolInterface:
        """Return the tool named ``name`` or raise."""
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(name)
        return tool

    def list(self) -> list[ToolInterface]:
        """Return all registered tools."""
        return list(self._tools.values())

    def list_info(self) -> list[ToolInfo]:
        """Return metadata for all registered tools."""
        return [t.info for t in self._tools.values()]

    def exists(self, name: str) -> bool:
        """Return whether a tool named ``name`` is registered."""
        return name in self._tools

    def find_by_category(self, category: ToolCategory) -> list[ToolInterface]:
        """Return all tools in ``category``."""
        return [t for t in self._tools.values() if t.category is category]

    async def list_available(
        self,
        *,
        policy: ToolPolicy,
        context: ToolDecisionContext | None = None,
        environments: set[ToolEnvironment] | None = None,
    ) -> list[ToolInfo]:
        """Return tool infos available to a caller under ``policy``/``context``.

        This is a *discovery* mechanism. It never authorizes anything: the
        executor re-evaluates :class:`~app.tools.policy.ToolPolicy` for every
        execution. See also :func:`_enforce_dynamic_restrictions`.

        ``environments`` optionally filters to tools whose supported
        environments intersect the requested set.
        """
        if policy is None:
            return []
        infos = self.list_info()
        infos = self._enforce_dynamic_restrictions(infos, context)
        if environments is not None:
            infos = [
                i
                for i in infos
                if i.supported_environments and environments.intersection(i.supported_environments)
            ]
        available = policy.list_available_for(infos, context)
        for info in available:
            await self._publish_discovered(info, context)
        return available

    def _enforce_dynamic_restrictions(
        self, infos: list[ToolInfo], context: ToolDecisionContext | None
    ) -> list[ToolInfo]:
        """Strip HIGH/CRITICAL tools from dynamic agents' discovery view.

        Dynamic agents never automatically receive HIGH- or CRITICAL-risk
        tools. Any explicit allow for a dynamic agent must come from trusted,
        preconfigured policy, never from the dynamic agent or the LLM.
        """
        if context is None or not context.agent_dynamic:
            return infos
        return [i for i in infos if i.risk_level not in (RiskLevel.HIGH, RiskLevel.CRITICAL)]

    async def _publish_discovered(
        self, info: ToolInfo, context: ToolDecisionContext | None
    ) -> None:
        if self._event_bus is None:
            return
        from app.events import Event, EventType

        task_id = context.task_id if context else None
        session_id = context.session_id if context else None
        agent_id = context.agent_id if context else None
        payload = {
            "tool": info.name,
            "category": info.category.value,
            "risk": info.risk_level.value,
            "agent_id": agent_id,
            "agent_dynamic": bool(context and context.agent_dynamic),
            "task_id": task_id,
            "session_id": session_id,
        }
        await self._event_bus.publish(
            Event.create(
                EventType.TOOL_DISCOVERED,
                task_id=task_id,
                session_id=session_id,
                agent_id=agent_id,
                payload=payload,
                metadata={"tool": info.name, "risk": info.risk_level.value},
            )
        )

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._tools
