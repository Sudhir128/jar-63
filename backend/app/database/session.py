"""SQLAlchemy engine, session factory, base model, and health-check.

No application-specific tables are created in this phase. Engines are created
on import but do not open connections until first use.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator

from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger("database")

__all__ = [
    "Base",
    "engine",
    "SessionLocal",
    "async_engine",
    "AsyncSessionLocal",
    "get_db",
    "get_async_db",
    "check_db_connection",
    "check_async_db_connection",
]


class Base(DeclarativeBase):
    """Declarative base for all ORM models (defined in later phases)."""

    pass


def _is_sqlite(url: str) -> bool:
    return url.startswith("sqlite")


def _build_engine():
    settings = get_settings()
    url = settings.database.effective_url
    kwargs: dict[str, object] = {
        "echo": settings.database.echo,
        "pool_pre_ping": not _is_sqlite(url),
        "future": True,
    }
    if not _is_sqlite(url):
        kwargs["pool_size"] = settings.database.pool_size
        kwargs["max_overflow"] = settings.database.max_overflow
    else:
        from sqlalchemy.pool import StaticPool

        kwargs["poolclass"] = StaticPool
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **kwargs)


def _to_async_url(url: str) -> str:
    if url.startswith("postgresql+psycopg://"):
        return url.replace("postgresql+psycopg://", "postgresql+asyncpg://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if url.startswith("sqlite://"):
        return url.replace("sqlite://", "sqlite+aiosqlite://", 1)
    return url


def _build_async_engine() -> AsyncEngine:
    settings = get_settings()
    url = _to_async_url(settings.database.effective_url)
    kwargs: dict[str, object] = {
        "echo": settings.database.echo,
        "pool_pre_ping": not _is_sqlite(url),
        "future": True,
    }
    if not _is_sqlite(url):
        kwargs["pool_size"] = settings.database.pool_size
        kwargs["max_overflow"] = settings.database.max_overflow
    else:
        # SQLite: use a single shared connection so :memory: persists across
        # sessions (required for tests and demos that write then read).
        from sqlalchemy.pool import StaticPool

        kwargs["poolclass"] = StaticPool
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_async_engine(url, **kwargs)


engine = _build_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

async_engine = _build_async_engine()
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
)


def get_db() -> Generator[Session]:
    """FastAPI dependency yielding a sync database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_async_db() -> AsyncGenerator[AsyncSession]:
    """FastAPI dependency yielding an async database session."""
    async with AsyncSessionLocal() as session:
        yield session


def check_db_connection() -> bool:
    """Return whether the database is reachable (sync health-check)."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:  # noqa: BLE001
        logger.bind(error=type(exc).__name__).warning("Database health-check failed: {}", str(exc))
        return False


async def check_async_db_connection() -> bool:
    """Return whether the database is reachable (async health-check)."""
    try:
        async with async_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:  # noqa: BLE001
        logger.bind(error=type(exc).__name__).warning(
            "Async database health-check failed: {}", str(exc)
        )
        return False
