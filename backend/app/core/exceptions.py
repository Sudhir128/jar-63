"""Domain-specific exception hierarchy."""

from __future__ import annotations

__all__ = [
    "JARError",
    "ConfigurationError",
    "AgentError",
    "AgentNotFoundError",
    "AgentAlreadyRegisteredError",
    "ToolError",
    "ToolNotFoundError",
    "ToolAlreadyRegisteredError",
    "EventBusError",
    "RuntimeError_",
]


class JARError(Exception):
    """Base class for all JAR-63 domain errors."""


class ConfigurationError(JARError):
    """Raised when configuration is invalid or incomplete."""


class AgentError(JARError):
    """Base class for agent-related errors."""


class AgentNotFoundError(AgentError):
    """Raised when a requested agent is not registered."""


class AgentAlreadyRegisteredError(AgentError):
    """Raised when registering an agent whose id already exists."""


class ToolError(JARError):
    """Base class for tool-related errors."""


class ToolNotFoundError(ToolError):
    """Raised when a requested tool is not registered."""


class ToolAlreadyRegisteredError(ToolError):
    """Raised when registering a tool whose name already exists."""


class EventBusError(JARError):
    """Raised for event bus operational failures."""


class RuntimeError_(JARError):
    """Raised for runtime orchestration failures.

    Aliased with a trailing underscore to avoid shadowing the builtin.
    """
