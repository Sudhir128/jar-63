"""Memory health checker (Phase 6).

Provides a non-fatal health check for the memory subsystem. Used by the
runtime at startup and by the ``/health`` and ``/api/v1/memory/status``
endpoints. The check never raises — it reports degraded states.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.logging import get_logger
from app.memory.manager import MemoryManager

logger = get_logger("memory.health")

__all__ = ["MemoryHealthChecker", "MemoryHealthSnapshot"]


@dataclass(frozen=True)
class MemoryHealthSnapshot:
    """Snapshot of memory subsystem health."""

    enabled: bool
    pg_ready: bool
    redis_ready: bool
    vector_ready: bool
    record_count: int
    status: str  # "ok" | "degraded" | "disabled"

    def to_api_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "enabled": self.enabled,
            "pg_ready": self.pg_ready,
            "redis_ready": self.redis_ready,
            "vector_ready": self.vector_ready,
            "record_count": self.record_count,
        }


class MemoryHealthChecker:
    """Non-fatal memory health checker."""

    def __init__(self, manager: MemoryManager) -> None:
        self._manager = manager
        self._snapshot: MemoryHealthSnapshot | None = None

    @property
    def snapshot(self) -> MemoryHealthSnapshot | None:
        """Return the most recent health snapshot, or ``None`` if never checked."""
        return self._snapshot

    async def check(self) -> MemoryHealthSnapshot:
        state = await self._manager.health_check()
        if not state.enabled:
            status = "disabled"
        elif state.pg_ready:
            status = (
                "ok"
                if (state.redis_ready or not self._manager.settings.redis_working_memory)
                else "degraded"
            )
        else:
            status = "degraded"
        snap = MemoryHealthSnapshot(
            enabled=state.enabled,
            pg_ready=state.pg_ready,
            redis_ready=state.redis_ready,
            vector_ready=state.vector_ready,
            record_count=state.record_count,
            status=status,
        )
        self._snapshot = snap
        logger.bind(
            event="memory.health.checked",
            status=status,
            records=snap.record_count,
        ).debug("Memory health: {}", status)
        return snap
