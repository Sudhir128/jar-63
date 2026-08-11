"""Agent interface, context, result, and capability definitions.

This is the common contract every future agent will implement. No concrete
agents are provided in this phase — only the interface and supporting models.
"""

from __future__ import annotations

import abc
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.core.identifiers import generate_id

__all__ = [
    "AgentCapability",
    "AgentStatus",
    "AgentInfo",
    "AgentContext",
    "AgentResult",
    "AgentInterface",
]


class AgentCapability(StrEnum):
    """Enumerated capabilities an agent may declare.

    Extensible in later phases; agents declare a subset relevant to them.
    """

    REASONING = "reasoning"
    PLANNING = "planning"
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    RESEARCH = "research"
    MATH = "math"
    FILESYSTEM = "filesystem"
    BROWSER = "browser"
    SHELL = "shell"
    VOICE = "voice"
    MEMORY = "memory"
    DEPLOYMENT = "deployment"
    COMMUNICATION = "communication"
    CUSTOM = "custom"


class AgentStatus(StrEnum):
    """Lifecycle status of an agent instance."""

    CREATED = "created"
    REGISTERED = "registered"
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    WAITING_FOR_CONFIRMATION = "waiting_for_confirmation"
    FAILED = "failed"
    UNREGISTERED = "unregistered"


class AgentInfo(BaseModel):
    """Static metadata describing an agent."""

    model_config = ConfigDict(frozen=True)

    agent_id: str
    name: str
    description: str = ""
    capabilities: set[AgentCapability] = Field(default_factory=set)
    version: str = "0.1.0"


class AgentContext(BaseModel):
    """Input passed to :meth:`AgentInterface.execute`."""

    model_config = ConfigDict(extra="allow")

    task_id: str = Field(default_factory=lambda: generate_id("task"))
    session_id: str | None = None
    input: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentResult(BaseModel):
    """Output returned by :meth:`AgentInterface.execute`."""

    model_config = ConfigDict(extra="allow")

    task_id: str
    agent_id: str
    status: AgentStatus = AgentStatus.COMPLETED
    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentInterface(abc.ABC):
    """Common interface/base class for all agents.

    Subclasses must implement :meth:`execute`. Lifecycle hooks have default
    no-op implementations and may be overridden as needed.
    """

    @property
    @abc.abstractmethod
    def info(self) -> AgentInfo:
        """Return the agent's static metadata."""

    @property
    def agent_id(self) -> str:
        return self.info.agent_id

    @property
    def name(self) -> str:
        return self.info.name

    @property
    def description(self) -> str:
        return self.info.description

    @property
    def capabilities(self) -> set[AgentCapability]:
        return self.info.capabilities

    @abc.abstractmethod
    async def execute(self, context: AgentContext) -> AgentResult:
        """Perform the agent's work asynchronously."""

    # --- Lifecycle hooks (overridable, default no-op) ---
    async def on_register(self) -> None:  # noqa: B027
        """Called once when the agent is registered with the registry."""

    async def on_unregister(self) -> None:  # noqa: B027
        """Called once when the agent is unregistered from the registry."""

    async def on_start(self, context: AgentContext) -> None:  # noqa: B027
        """Called before :meth:`execute` runs."""

    async def on_complete(self, result: AgentResult) -> None:  # noqa: B027
        """Called after :meth:`execute` succeeds."""

    async def on_error(self, error: BaseException) -> None:  # noqa: B027
        """Called when :meth:`execute` raises."""
