"""Redis-backed working memory and cache (Phase 6).

Working memory holds the *current* loop/task context — recent observations,
intermediate results, and short-lived scratch data — with a TTL. It is
distinct from the durable PostgreSQL store. When Redis is unavailable, the
manager treats working memory as empty (graceful degradation) rather than
failing the request.

The store stores JSON-serialized :class:`MemoryRecord` objects under
keys namespaced by session/task. A separate cache namespace memoizes
retrieval results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.core.logging import get_logger
from app.memory.models import MemoryRecord

if TYPE_CHECKING:
    from app.config import MemorySettings

logger = get_logger("memory.working")

__all__ = ["RedisWorkingMemoryStore"]

_KEY_PREFIX = "jar63:working"
_CACHE_PREFIX = "jar63:memcache"


class RedisWorkingMemoryStore:
    """Working memory + cache backed by Redis.

    All methods are fault-tolerant: a Redis error is logged and the method
    returns an empty/neutral result rather than raising. Callers never crash
    because working memory is unavailable.
    """

    def __init__(
        self,
        client: Redis,
        settings: MemorySettings | None = None,
    ) -> None:
        self._client = client
        self._settings = settings or _default_settings()
        self._ttl = self._settings.working_memory_ttl_seconds
        self._cache_ttl = self._settings.cache_ttl_seconds

    # --- Working memory ---
    async def add(self, record: MemoryRecord) -> bool:
        """Add a memory to the working set for its session/task."""
        key = self._scope_key(record)
        try:
            await self._client.sadd(key, record.model_dump_json())
            await self._client.expire(key, self._ttl)
            return True
        except RedisError as exc:
            logger.bind(error=type(exc).__name__).warning("Working memory add failed: {}", str(exc))
            return False

    async def list(
        self,
        *,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> list[MemoryRecord]:
        """List working memories for a session or task."""
        key = self._scope_key_from(session_id=session_id, task_id=task_id)
        if key is None:
            return []
        try:
            raw = await self._client.smembers(key)
        except RedisError as exc:
            logger.bind(error=type(exc).__name__).warning(
                "Working memory list failed: {}", str(exc)
            )
            return []
        records = []
        for item in raw:
            try:
                records.append(MemoryRecord.model_validate_json(item))
            except Exception:  # noqa: BLE001 - skip corrupt entries
                continue
        return records

    async def clear(
        self,
        *,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> int:
        """Clear working memory for a session/task. Return items removed."""
        key = self._scope_key_from(session_id=session_id, task_id=task_id)
        if key is None:
            return 0
        try:
            return int(await self._client.delete(key))
        except RedisError as exc:
            logger.bind(error=type(exc).__name__).warning(
                "Working memory clear failed: {}", str(exc)
            )
            return 0

    # --- Retrieval cache ---
    async def get_cache(self, cache_key: str) -> str | None:
        try:
            return await self._client.get(f"{_CACHE_PREFIX}:{cache_key}")
        except RedisError as exc:
            logger.bind(error=type(exc).__name__).warning("Memory cache get failed: {}", str(exc))
            return None

    async def set_cache(self, cache_key: str, value: str) -> bool:
        try:
            await self._client.set(f"{_CACHE_PREFIX}:{cache_key}", value, ex=self._cache_ttl)
            return True
        except RedisError as exc:
            logger.bind(error=type(exc).__name__).warning("Memory cache set failed: {}", str(exc))
            return False

    async def invalidate_cache(self, cache_key: str) -> bool:
        try:
            return bool(await self._client.delete(f"{_CACHE_PREFIX}:{cache_key}"))
        except RedisError as exc:
            logger.bind(error=type(exc).__name__).warning(
                "Memory cache invalidate failed: {}", str(exc)
            )
            return False

    # --- Internal helpers ---
    def _scope_key(self, record: MemoryRecord) -> str:
        return self._scope_key_from(session_id=record.session_id, task_id=record.task_id)

    def _scope_key_from(self, *, session_id: str | None, task_id: str | None) -> str | None:
        if task_id:
            return f"{_KEY_PREFIX}:task:{task_id}"
        if session_id:
            return f"{_KEY_PREFIX}:session:{session_id}"
        return None


def _default_settings() -> MemorySettings:
    from app.config import MemorySettings

    return MemorySettings()
