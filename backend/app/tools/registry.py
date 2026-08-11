"""Tool registry: register, unregister, retrieve, list, and check existence."""

from __future__ import annotations

from app.core.exceptions import ToolAlreadyRegisteredError, ToolNotFoundError
from app.core.logging import get_logger
from app.tools.interface import ToolCategory, ToolInfo, ToolInterface

logger = get_logger("tools.registry")

__all__ = ["ToolRegistry"]


class ToolRegistry:
    """In-process registry of tools.

    Depends only on :class:`ToolInterface`, never on specific tool
    implementations.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolInterface] = {}

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

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._tools
