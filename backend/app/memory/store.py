"""Memory store abstraction and PostgreSQL implementation.

The :class:`MemoryStore` ABC defines the persistence contract. The concrete
:class:`PostgreSQLStore` uses async SQLAlchemy against PostgreSQL (and falls
back to SQLite-compatible operations for the test suite, which uses an
in-memory SQLite database — no pgvector required).

The store boundary translates between SQLAlchemy ORM models
(:mod:`app.models.memory`) and the Pydantic domain models
(:mod:`app.memory.models`). The domain layer never sees ORM objects.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from sqlalchemy import delete as sa_delete
from sqlalchemy import func, select, update

from app.core.identifiers import utc_now
from app.core.logging import get_logger
from app.database import AsyncSessionLocal
from app.memory.models import MemoryRecord
from app.memory.types import MemoryType
from app.models.memory import MemoryModel

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from app.memory.models import MemorySearchQuery, MemorySearchResult

logger = get_logger("memory.store")

__all__ = ["MemoryStore", "PostgreSQLStore"]


class MemoryStore(abc.ABC):
    """Persistence contract for durable structured memories."""

    @abc.abstractmethod
    async def create(self, record: MemoryRecord) -> MemoryRecord:
        """Insert a new memory record."""

    @abc.abstractmethod
    async def get(self, memory_id: str) -> MemoryRecord | None:
        """Retrieve a memory by ID, updating its access metadata."""

    @abc.abstractmethod
    async def update(self, memory_id: str, **fields: object) -> MemoryRecord | None:
        """Update fields on an existing memory and return the new snapshot."""

    @abc.abstractmethod
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory. Return whether a row was removed."""

    @abc.abstractmethod
    async def search(self, query: MemorySearchQuery) -> list[MemorySearchResult]:
        """Structured (metadata) search over memories."""

    @abc.abstractmethod
    async def list_by(
        self,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
        memory_types: list[MemoryType] | None = None,
        limit: int = 50,
    ) -> list[MemoryRecord]:
        """List memories filtered by scope, ordered by recency."""

    @abc.abstractmethod
    async def count(
        self,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> int:
        """Count stored memories (optionally scoped)."""

    @abc.abstractmethod
    async def delete_expired(self) -> int:
        """Delete expired memories. Return the number removed."""

    async def close(self) -> None:  # noqa: B027
        """Release resources (optional override)."""


class PostgreSQLStore(MemoryStore):
    """PostgreSQL-backed durable memory store.

    Works against PostgreSQL in production and SQLite in tests (both through
    async SQLAlchemy). Embeddings are stored as JSON arrays so no pgvector
    extension is required.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
    ) -> None:
        self._session_factory = session_factory or AsyncSessionLocal

    async def create(self, record: MemoryRecord) -> MemoryRecord:
        async with self._session_factory() as session:
            row = MemoryModel(
                memory_id=record.memory_id,
                memory_type=record.memory_type.value,
                content=record.content,
                summary=record.summary,
                user_id=record.user_id,
                session_id=record.session_id,
                task_id=record.task_id,
                agent_id=record.agent_id,
                source=record.source.value,
                importance=record.importance,
                confidence=record.confidence,
                retention_policy=record.retention_policy.value,
                embedding=None,
                created_at=record.created_at,
                updated_at=record.updated_at,
                last_accessed_at=record.last_accessed_at,
                expires_at=record.expires_at,
                access_count=record.access_count,
                metadata_=dict(record.metadata),
            )
            session.add(row)
            await session.commit()
            return record

    async def get(self, memory_id: str) -> MemoryRecord | None:
        async with self._session_factory() as session:
            row = await session.get(MemoryModel, memory_id)
            if row is None:
                return None
            now = utc_now()
            await session.execute(
                update(MemoryModel)
                .where(MemoryModel.memory_id == memory_id)
                .values(last_accessed_at=now, access_count=row.access_count + 1)
            )
            await session.commit()
            return _row_to_record(row)

    async def update(self, memory_id: str, **fields: object) -> MemoryRecord | None:
        if not fields:
            return await self.get(memory_id)
        allowed = {
            "content",
            "summary",
            "importance",
            "confidence",
            "expires_at",
            "metadata",
        }
        clean: dict[str, object] = {}
        for k, v in fields.items():
            if k in allowed:
                clean[k if k != "metadata" else "metadata_"] = v
        if not clean:
            return await self.get(memory_id)
        clean["updated_at"] = utc_now()
        async with self._session_factory() as session:
            row = await session.get(MemoryModel, memory_id)
            if row is None:
                return None
            await session.execute(
                update(MemoryModel).where(MemoryModel.memory_id == memory_id).values(**clean)
            )
            await session.commit()
            row = await session.get(MemoryModel, memory_id)
            return _row_to_record(row) if row else None

    async def delete(self, memory_id: str) -> bool:
        async with self._session_factory() as session:
            result = await session.execute(
                sa_delete(MemoryModel).where(MemoryModel.memory_id == memory_id)
            )
            await session.commit()
            return (result.rowcount or 0) > 0

    async def search(self, query: MemorySearchQuery) -> list[MemorySearchResult]:
        from app.memory.models import MemorySearchResult

        stmt = select(MemoryModel)
        stmt = _apply_scope(stmt, query)
        if query.memory_types:
            stmt = stmt.where(MemoryModel.memory_type.in_([t.value for t in query.memory_types]))
        if not query.include_expired:
            stmt = stmt.where(
                (MemoryModel.expires_at.is_(None)) | (MemoryModel.expires_at > utc_now())
            )
        stmt = stmt.where(MemoryModel.importance >= query.min_importance).where(
            MemoryModel.confidence >= query.min_confidence
        )
        # Structured search: case-insensitive substring match on content/summary.
        if query.query:
            like = f"%{query.query.lower()}%"
            stmt = stmt.where(
                func.lower(MemoryModel.content).like(like)
                | func.lower(MemoryModel.summary).like(like)
            )
        stmt = stmt.order_by(MemoryModel.importance.desc(), MemoryModel.created_at.desc()).limit(
            query.top_k
        )
        async with self._session_factory() as session:
            rows = (await session.execute(stmt)).scalars().all()
            results = []
            for row in rows:
                score = _structured_score(row, query.query)
                results.append(
                    MemorySearchResult(
                        memory=_row_to_record(row),
                        score=score,
                        matched_by="structured",
                    )
                )
            return results

    async def list_by(
        self,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
        memory_types: list[MemoryType] | None = None,
        limit: int = 50,
    ) -> list[MemoryRecord]:
        stmt = select(MemoryModel)
        if user_id is not None:
            stmt = stmt.where(MemoryModel.user_id == user_id)
        if session_id is not None:
            stmt = stmt.where(MemoryModel.session_id == session_id)
        if task_id is not None:
            stmt = stmt.where(MemoryModel.task_id == task_id)
        if memory_types:
            stmt = stmt.where(MemoryModel.memory_type.in_([t.value for t in memory_types]))
        stmt = stmt.order_by(MemoryModel.created_at.desc()).limit(limit)
        async with self._session_factory() as session:
            rows = (await session.execute(stmt)).scalars().all()
            return [_row_to_record(r) for r in rows]

    async def count(
        self,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> int:
        stmt = select(func.count()).select_from(MemoryModel)
        if user_id is not None:
            stmt = stmt.where(MemoryModel.user_id == user_id)
        if session_id is not None:
            stmt = stmt.where(MemoryModel.session_id == session_id)
        async with self._session_factory() as session:
            result = await session.execute(stmt)
            return int(result.scalar() or 0)

    async def delete_expired(self) -> int:
        async with self._session_factory() as session:
            result = await session.execute(
                sa_delete(MemoryModel).where(
                    (MemoryModel.expires_at.is_not(None)) & (MemoryModel.expires_at <= utc_now())
                )
            )
            await session.commit()
            return int(result.rowcount or 0)


def _apply_scope(stmt, query: MemorySearchQuery):
    """Apply user/session/task/agent scope filters to a statement."""
    if query.user_id is not None:
        stmt = stmt.where(MemoryModel.user_id == query.user_id)
    if query.session_id is not None:
        stmt = stmt.where(MemoryModel.session_id == query.session_id)
    if query.task_id is not None:
        stmt = stmt.where(MemoryModel.task_id == query.task_id)
    if query.agent_id is not None:
        stmt = stmt.where(MemoryModel.agent_id == query.agent_id)
    return stmt


def _structured_score(row: MemoryModel, query_text: str) -> float:
    """Deterministic relevance score for a structured match."""
    base = 0.4
    if not query_text:
        return base
    lower_q = query_text.lower()
    lower_content = (row.content or "").lower()
    if lower_q in lower_content:
        base += 0.3
    if lower_q in (row.summary or "").lower():
        base += 0.1
    base += 0.2 * row.importance
    return min(base, 1.0)


def _row_to_record(row: MemoryModel) -> MemoryRecord:
    """Translate an ORM row into a :class:`MemoryRecord`."""
    return MemoryRecord(
        memory_id=row.memory_id,
        memory_type=MemoryType(row.memory_type),
        content=row.content,
        summary=row.summary or "",
        user_id=row.user_id,
        session_id=row.session_id,
        task_id=row.task_id,
        agent_id=row.agent_id,
        source=_safe_source(row.source),
        importance=float(row.importance or 0.5),
        confidence=float(row.confidence or 0.5),
        retention_policy=_safe_retention(row.retention_policy),
        created_at=row.created_at,
        updated_at=row.updated_at,
        last_accessed_at=row.last_accessed_at,
        expires_at=row.expires_at,
        access_count=int(row.access_count or 0),
        metadata=dict(row.metadata_ or {}),
    )


def _safe_source(value: str | None):
    from app.memory.types import MemorySource

    if not value:
        return MemorySource.SYSTEM
    try:
        return MemorySource(value)
    except ValueError:
        return MemorySource.SYSTEM


def _safe_retention(value: str | None):
    from app.memory.types import RetentionPolicy

    if not value:
        return RetentionPolicy.TTL
    try:
        return RetentionPolicy(value)
    except ValueError:
        return RetentionPolicy.TTL
