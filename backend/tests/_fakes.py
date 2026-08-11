"""Minimal concrete agent and tool implementations used by tests."""

from __future__ import annotations

from app.agents.interface import (
    AgentCapability,
    AgentContext,
    AgentInfo,
    AgentInterface,
    AgentResult,
    AgentStatus,
)
from app.tools.interface import (
    ToolCategory,
    ToolContext,
    ToolInfo,
    ToolInterface,
    ToolResult,
)


class EchoAgent(AgentInterface):
    """A test agent that echoes its input."""

    def __init__(
        self,
        agent_id: str = "echo",
        name: str = "Echo",
        capabilities: set[AgentCapability] | None = None,
    ) -> None:
        self._info = AgentInfo(
            agent_id=agent_id,
            name=name,
            description="Echoes input back.",
            capabilities=capabilities or {AgentCapability.REASONING},
        )

    @property
    def info(self) -> AgentInfo:
        return self._info

    async def execute(self, context: AgentContext) -> AgentResult:
        return AgentResult(
            task_id=context.task_id,
            agent_id=self.agent_id,
            status=AgentStatus.COMPLETED,
            output=context.input,
        )


class ErrorAgent(AgentInterface):
    """A test agent that always raises."""

    def __init__(self, agent_id: str = "error") -> None:
        self._info = AgentInfo(agent_id=agent_id, name="Error")

    @property
    def info(self) -> AgentInfo:
        return self._info

    async def execute(self, context: AgentContext) -> AgentResult:
        raise RuntimeError("intentional failure")


class EchoTool(ToolInterface):
    """A test tool that echoes its arguments."""

    def __init__(self, name: str = "echo") -> None:
        self._info = ToolInfo(
            name=name, description="Echoes arguments.", category=ToolCategory.PYTHON
        )

    @property
    def info(self) -> ToolInfo:
        return self._info

    async def execute(self, context: ToolContext) -> ToolResult:
        return ToolResult(
            invocation_id=context.invocation_id, name=self.name, output=context.arguments
        )
