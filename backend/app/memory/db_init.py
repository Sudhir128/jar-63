"""Database table initialization for the memory subsystem (Phase 6).

Creates the memory-related tables (``memories``, ``conversation_messages``,
``memory_relations``) via ``Base.metadata.create_all`` against the configured
async engine. This is idempotent and safe for dev/test. Production should
prefer Alembic migrations, but this ensures the tables exist without a
separate migration step during early development.
"""

from __future__ import annotations

from app.database import Base, async_engine

__all__ = ["init_memory_tables"]


async def init_memory_tables() -> None:
    """Create memory tables if they do not exist (idempotent)."""
    # Import models so their tables are registered on Base.metadata.
    import app.models.memory  # noqa: F401

    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
