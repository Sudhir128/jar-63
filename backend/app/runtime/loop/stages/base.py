"""Stage base interface.

Every stage is an async callable over a :class:`LoopContext`. Stages produce
typed result objects and publish their own events. No stage is coupled to a
particular agent, tool, or LLM — intelligence is injected through the stage
implementation.
"""

from __future__ import annotations

import abc

from app.runtime.loop.loop_context import LoopContext

__all__ = ["LoopStage", "StageResult"]


class StageResult:
    """Marker base for typed stage results (used for documentation typing)."""

    pass


class LoopStage(abc.ABC):
    """Abstract stage contract for the Universal Loop Engine."""

    name: str = "stage"

    @abc.abstractmethod
    async def run(self, context: LoopContext) -> LoopContext:
        """Execute the stage, evolving ``context.state`` and returning it.

        Stages mutate state only through :meth:`LoopContext.update_state`
        (functional updates). They publish events through
        ``context.event_bus``.
        """
