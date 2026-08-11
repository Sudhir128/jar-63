"""Agents package: interface and registry."""

from app.agents.interface import (
    AgentCapability,
    AgentContext,
    AgentInfo,
    AgentInterface,
    AgentResult,
    AgentStatus,
)
from app.agents.registry import AgentRegistry
from app.core.exceptions import AgentAlreadyRegisteredError, AgentError, AgentNotFoundError

__all__ = [
    "AgentAlreadyRegisteredError",
    "AgentCapability",
    "AgentContext",
    "AgentError",
    "AgentInfo",
    "AgentInterface",
    "AgentNotFoundError",
    "AgentRegistry",
    "AgentResult",
    "AgentStatus",
]
