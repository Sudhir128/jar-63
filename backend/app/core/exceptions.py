"""Domain-specific exception hierarchy."""

from __future__ import annotations

__all__ = [
    "JARError",
    "ConfigurationError",
    "AgentError",
    "AgentNotFoundError",
    "AgentAlreadyRegisteredError",
    "AgentDefinitionError",
    "AgentDefinitionNotFoundError",
    "AgentDefinitionAlreadyExistsError",
    "AgentDefinitionValidationError",
    "AgentNotDispatchableError",
    "AgentRoutingError",
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


class AgentDefinitionError(AgentError):
    """Base class for persistent agent definition errors."""


class AgentDefinitionNotFoundError(AgentDefinitionError):
    """Raised when a requested agent definition is not in the catalog."""


class AgentDefinitionAlreadyExistsError(AgentDefinitionError):
    """Raised when registering a definition whose id already exists."""


class AgentDefinitionValidationError(AgentDefinitionError):
    """Raised when an agent definition/spec fails validation."""


class AgentNotDispatchableError(AgentError):
    """Raised when attempting to dispatch a non-active agent definition."""


class AgentRoutingError(AgentError):
    """Raised when agent routing cannot produce a decision."""


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
