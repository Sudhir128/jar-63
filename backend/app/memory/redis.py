"""Redis foundation: connection abstraction and health-check.

No caching or queueing logic is implemented yet; only connection management.
"""

from __future__ import annotations

from redis.asyncio import Redis, from_url
from redis.exceptions import RedisError

from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger("redis")

__all__ = ["RedisClient", "get_redis", "check_redis_connection", "close_redis"]


def get_redis() -> Redis:
    """Return a shared async Redis client built from configuration."""
    settings = get_settings()
    return from_url(
        settings.redis.effective_url,
        decode_responses=True,
        socket_connect_timeout=5,
        retry_on_timeout=True,
    )


# A module-level lazy client for simple sharing within a process.
_client: Redis | None = None


def get_shared_redis() -> Redis:
    """Return a process-wide shared Redis client (lazily created)."""
    global _client
    if _client is None:
        _client = get_redis()
    return _client


async def check_redis_connection(client: Redis | None = None) -> bool:
    """Return whether Redis is reachable via ``PING``."""
    owns_client = client is None
    client = client or get_redis()
    try:
        return bool(await client.ping())
    except RedisError as exc:
        logger.bind(error=type(exc).__name__).warning("Redis health-check failed: {}", str(exc))
        return False
    finally:
        if owns_client:
            await client.aclose()


async def close_redis() -> None:
    """Close the shared Redis client if one was created."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


# Type alias for clarity in dependency injection.
RedisClient = Redis
