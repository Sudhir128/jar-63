"""Memory-aware discover stage (Phase 6).

Wraps the deterministic :class:`DefaultDiscoverStage`, retrieving a bounded
:class:`MemoryContext` for the task goal and stashing it on the loop state so
the PLAN stage (and future LLM planner) can use it.

The stage is opt-in: when no :class:`MemoryManager` is wired (or memory is
disabled), it behaves exactly like the default discover stage. This keeps the
loop engine backward-compatible with Phase 1-5.

Memories are treated as **untrusted contextual information**: they are
labeled clearly and must never override system policy. The memory context is
stored under ``task.metadata["memory_context"]`` for later stages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.core.logging import get_logger
from app.runtime.loop.loop_context import LoopContext
from app.runtime.loop.loop_state import StageStatus
from app.runtime.loop.stages.base import LoopStage
from app.runtime.loop.stages.discover import (
    DefaultDiscoverStage,
    DiscoveryResult,
    DiscoverStage,
)

if TYPE_CHECKING:
    from app.memory.manager import MemoryManager

logger = get_logger("loop.discover.memory")

__all__ = ["MemoryDiscoverStage"]


class MemoryDiscoverStage(DiscoverStage):
    """Discover stage enriched with persistent memory retrieval.

    Delegates the base discovery to :class:`DefaultDiscoverStage`, then
    retrieves a bounded memory context for the task goal and records it on
    the loop state for downstream stages.
    """

    def __init__(
        self,
        memory_manager: MemoryManager | None = None,
        inner: DiscoverStage | None = None,
    ) -> None:
        self._memory = memory_manager
        self._inner = inner or DefaultDiscoverStage()

    name = "discover"

    async def discover(self, context: LoopContext) -> DiscoveryResult:
        base = await self._inner.discover(context)
        memory_context = await self._retrieve_memory(context, base)
        # Stash memory context on task metadata for the planner.
        context.task.metadata["memory_context"] = memory_context.model_dump(mode="json")
        # Record that memory was consulted (count only — no content).
        context.task.metadata.setdefault("discovery_meta", {})[
            "memory_count"
        ] = len(memory_context.memories)
        return base

    async def _retrieve_memory(
        self, context: LoopContext, base: DiscoveryResult
    ) -> object:
        """Retrieve a bounded memory context, or an empty one if unavailable."""
        from app.memory.models import MemoryContext

        if self._memory is None or not self._memory.is_enabled:
            return MemoryContext(
                query=base.goal,
                session_id=base.session_id,
            )
        try:
            return await self._memory.retrieve_context(
                base.goal,
                session_id=base.session_id,
                task_id=context.task_id,
                agent_id=base.agent_id,
            )
        except Exception as exc:  # noqa: BLE001 - memory must never break the loop
            logger.bind(
                event="memory.discover.error",
                error=type(exc).__name__,
                task_id=context.task_id,
            ).warning("Memory retrieval in discover failed (non-fatal): {}", str(exc))
            return MemoryContext(query=base.goal, session_id=base.session_id, task_id=context.task_id)
