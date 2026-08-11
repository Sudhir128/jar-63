"""Discover stage.

Gathers the available context: the task goal, session info, available agents
and tools, existing task state, success criteria, and relevant metadata.

Phase 1 implements deterministic, non-LLM discovery. The architecture allows
an LLM-powered discovery stage to replace :class:`DefaultDiscoverStage` later
without changing the loop engine.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.core.logging import get_logger
from app.runtime.loop.loop_context import LoopContext
from app.runtime.loop.loop_state import StageStatus
from app.runtime.loop.stages.base import LoopStage

logger = get_logger("loop.discover")

__all__ = ["DiscoveryResult", "DiscoverStage", "DefaultDiscoverStage"]


class DiscoveryResult(BaseModel):
    """Typed output of the Discover stage."""

    model_config = ConfigDict(frozen=True, extra="allow")

    goal: str = ""
    session_id: str | None = None
    success_criteria: list[str] = Field(default_factory=list)
    available_agents: list[str] = Field(default_factory=list)
    available_tools: list[str] = Field(default_factory=list)
    agent_id: str | None = None
    expected_output: object | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class DiscoverStage(LoopStage):
    """Abstract discover stage contract."""

    name = "discover"

    async def run(self, context: LoopContext) -> LoopContext:
        result = await self.discover(context)
        context.update_state(current_stage=StageStatus.DISCOVER)
        # Stash discovery on the task metadata for later stages.
        context.task.metadata["discovery"] = result.model_dump()
        return context

    async def discover(self, context: LoopContext) -> DiscoveryResult:
        """Produce a typed discovery result."""


class DefaultDiscoverStage(DiscoverStage):
    """Deterministic discovery: reads task fields and registry contents."""

    async def discover(self, context: LoopContext) -> DiscoveryResult:
        task = context.task
        agents = [a.agent_id for a in context.agent_registry.list()]
        tools = [t.name for t in context.tool_registry.list()]
        criteria = list(context.state.success_criteria)
        expected_output = context.stage_config.get("expected_output")
        return DiscoveryResult(
            goal=context.state.goal,
            session_id=context.session_id,
            success_criteria=criteria,
            available_agents=agents,
            available_tools=tools,
            agent_id=task.agent_id,
            expected_output=expected_output,
            metadata=dict(task.metadata or {}),
        )
