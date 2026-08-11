"""Memory manager: the coordinator for the memory subsystem (Phase 6).

The :class:`MemoryManager` is the single entry point the rest of the system
uses to read and write memory. It composes:

* :class:`MemoryStore`         — durable PostgreSQL records
* :class:`RedisWorkingMemoryStore` — short-lived working memory + cache
* :class:`VectorMemoryStore`   — optional semantic retrieval
* :class:`EmbeddingProvider`   — local-first embeddings
* :class:`MemoryWritePolicy`   — privacy + duplicate guardrails
* :class:`MemoryRetriever`     — hybrid retrieval + bounded context packing
* :class:`ConversationStore`   — conversation history
* :class:`MemoryConsolidator`  — growth control

Design rules:
* The manager never exposes raw store objects to callers — it returns domain
  models (:class:`MemoryRecord`, :class:`MemoryContext`).
* Every write goes through the write policy (privacy + dedup).
* Every successful write publishes a ``MEMORY_*`` event (no content).
* The manager degrades gracefully: if Redis/vector is unavailable, retrieval
  still works from PostgreSQL; if PostgreSQL is unavailable, reads return
  empty and writes are logged-but-not-stored (never crash the runtime).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.config import MemorySettings
from app.core.logging import get_logger
from app.events import Event, EventBus, EventType
from app.memory.consolidation import (
    BasicMemoryConsolidator,
    ConsolidationReport,
    MemoryConsolidator,
)
from app.memory.conversation import ConversationStore, ConversationSummarizer
from app.memory.embeddings import EmbeddingProvider, HashingEmbeddingProvider
from app.memory.models import ConversationMessage, MemoryContext, MemoryRecord
from app.memory.policy import MemoryWritePolicy, WriteDecision
from app.memory.retriever import MemoryRetriever
from app.memory.store import MemoryStore, PostgreSQLStore
from app.memory.types import MemorySource, MemoryType, MemoryWriteDecision
from app.memory.vector import InMemoryVectorStore, VectorMemoryStore
from app.memory.working_memory import RedisWorkingMemoryStore

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = get_logger("memory.manager")

__all__ = ["MemoryManager", "MemoryManagerState"]


@dataclass
class MemoryManagerState:
    """Snapshot of memory subsystem availability."""

    enabled: bool = False
    pg_ready: bool = False
    redis_ready: bool = False
    vector_ready: bool = False
    record_count: int = 0


@dataclass
class MemoryManager:
    """Coordinator for the persistent memory subsystem."""

    settings: MemorySettings
    event_bus: EventBus
    store: MemoryStore
    working_memory: RedisWorkingMemoryStore | None = None
    vector_store: VectorMemoryStore | None = None
    embedding_provider: EmbeddingProvider | None = None
    write_policy: MemoryWritePolicy = field(default_factory=MemoryWritePolicy)
    retriever: MemoryRetriever | None = None
    conversation_store: ConversationStore | None = None
    consolidator: MemoryConsolidator | None = None
    _state: MemoryManagerState = field(default_factory=MemoryManagerState)

    def __post_init__(self) -> None:
        if self.retriever is None:
            self.retriever = MemoryRetriever(
                store=self.store,
                vector_store=self.vector_store if self.settings.vector_enabled else None,
                settings=self.settings,
            )
        if self.consolidator is None:
            self.consolidator = BasicMemoryConsolidator(self.store)

    # --- Lifecycle ---
    async def start(self) -> None:
        self._state.enabled = self.settings.enabled
        self._state.pg_ready = self.settings.pg_persistence
        self._state.redis_ready = self.working_memory is not None
        self._state.vector_ready = self.settings.vector_enabled and self.vector_store is not None
        logger.bind(
            event="memory.started",
            pg=self._state.pg_ready,
            redis=self._state.redis_ready,
            vector=self._state.vector_ready,
        ).info("Memory manager started")

    async def shutdown(self) -> None:
        self._state.enabled = False
        logger.bind(event="memory.stopped").info("Memory manager stopped")

    @property
    def state(self) -> MemoryManagerState:
        return self._state

    @property
    def is_enabled(self) -> bool:
        return self._state.enabled

    # --- Writes ---
    async def add(
        self,
        *,
        content: str,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        user_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
        agent_id: str | None = None,
        source: MemorySource = MemorySource.SYSTEM,
        importance: float = 0.5,
        confidence: float = 0.5,
        summary: str = "",
        metadata: dict | None = None,
    ) -> MemoryRecord | None:
        """Create a memory through the write policy.

        Returns the stored record, or ``None`` if the write was rejected
        (privacy/duplicate) or persistence failed.
        """
        if not self._state.enabled:
            return None
        record = MemoryRecord(
            memory_type=memory_type,
            content=content,
            summary=summary,
            user_id=user_id,
            session_id=session_id,
            task_id=task_id,
            agent_id=agent_id,
            source=source,
            importance=importance,
            confidence=confidence,
            metadata=metadata or {},
        )
        if memory_type.is_long_lived is False:
            record = record.with_default_expiry()

        # Existing records for dedup (scoped).
        existing = await self._safe_list_for_dedup(record)

        decision = self.write_policy.evaluate(record, existing)
        return await self._apply_write_decision(record, decision)

    async def _apply_write_decision(
        self, record: MemoryRecord, decision: WriteDecision
    ) -> MemoryRecord | None:
        if decision.decision is MemoryWriteDecision.IGNORE:
            await self._publish(
                EventType.MEMORY_WRITE_REJECTED,
                memory_type=record.memory_type.value,
                source=record.source.value,
                reason=decision.reason,
                task_id=record.task_id,
                session_id=record.session_id,
            )
            logger.bind(event="memory.write.rejected", reason=decision.reason).debug(
                "Memory write rejected: {}", decision.reason
            )
            return None
        if decision.decision is MemoryWriteDecision.UPDATE and decision.existing_id:
            updated = await self.store.update(
                decision.existing_id,
                content=record.content,
                summary=record.summary or None,
                importance=record.importance,
                confidence=record.confidence,
                metadata=record.metadata,
            )
            if (
                updated is not None
                and self.vector_store is not None
                and self.settings.vector_enabled
            ):
                await self._safe_vector_upsert(updated)
            if updated is not None:
                await self._publish(
                    EventType.MEMORY_UPDATED,
                    memory_id=updated.memory_id,
                    memory_type=updated.memory_type.value,
                    task_id=updated.task_id,
                    session_id=updated.session_id,
                )
            return updated
        # STORE or EXPIRE → persist a new record.
        stored = await self._safe_create(record)
        if stored is None:
            return None
        if self.vector_store is not None and self.settings.vector_enabled:
            await self._safe_vector_upsert(stored)
        if self.working_memory is not None and stored.memory_type is MemoryType.WORKING:
            await self.working_memory.add(stored)
        await self._publish(
            EventType.MEMORY_CREATED,
            memory_id=stored.memory_id,
            memory_type=stored.memory_type.value,
            source=stored.source.value,
            task_id=stored.task_id,
            session_id=stored.session_id,
        )
        return stored

    async def _safe_create(self, record: MemoryRecord) -> MemoryRecord | None:
        try:
            return await self.store.create(record)
        except Exception as exc:  # noqa: BLE001 - never crash on persistence
            logger.bind(event="memory.create.error", error=type(exc).__name__).warning(
                "Memory create failed: {}", str(exc)
            )
            return None

    async def _safe_list_for_dedup(self, record: MemoryRecord) -> list[MemoryRecord]:
        try:
            return await self.store.list_by(
                user_id=record.user_id,
                session_id=record.session_id,
                task_id=record.task_id,
                memory_types=[record.memory_type],
                limit=50,
            )
        except Exception:  # noqa: BLE001
            return []

    async def _safe_vector_upsert(self, record: MemoryRecord) -> None:
        if self.vector_store is None or self.embedding_provider is None:
            return
        try:
            await self.vector_store.upsert(
                record.memory_id,
                record.content,
                metadata={
                    "user_id": record.user_id,
                    "session_id": record.session_id,
                    "task_id": record.task_id,
                    "memory_type": record.memory_type.value,
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.bind(event="memory.vector.upsert.error", error=type(exc).__name__).warning(
                "Vector upsert failed: {}", str(exc)
            )

    # --- Reads ---
    async def get(self, memory_id: str) -> MemoryRecord | None:
        try:
            return await self.store.get(memory_id)
        except Exception as exc:  # noqa: BLE001
            logger.bind(event="memory.get.error", error=type(exc).__name__).warning(
                "Memory get failed: {}", str(exc)
            )
            return None

    async def retrieve_context(
        self,
        query: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
        agent_id: str | None = None,
        memory_types: list[MemoryType] | None = None,
    ) -> MemoryContext:
        """Retrieve a bounded memory context for the loop/agent."""
        if not self._state.enabled or self.retriever is None:
            return MemoryContext(
                query=query, user_id=user_id, session_id=session_id, task_id=task_id
            )
        try:
            ctx = await self.retriever.retrieve_context(
                query,
                user_id=user_id,
                session_id=session_id,
                task_id=task_id,
                agent_id=agent_id,
                memory_types=memory_types,
            )
            await self._publish(
                EventType.MEMORY_RETRIEVED,
                task_id=task_id,
                session_id=session_id,
                payload={
                    "count": len(ctx.memories),
                    "total_available": ctx.total_available,
                    "truncated": ctx.truncated,
                },
            )
            return ctx
        except Exception as exc:  # noqa: BLE001
            logger.bind(event="memory.retrieve.error", error=type(exc).__name__).warning(
                "Memory retrieval failed: {}", str(exc)
            )
            await self._publish(
                EventType.MEMORY_SEARCH_FAILED,
                task_id=task_id,
                session_id=session_id,
                payload={"error": type(exc).__name__},
            )
            return MemoryContext(
                query=query, user_id=user_id, session_id=session_id, task_id=task_id
            )

    # --- Updates / Deletes ---
    async def update(self, memory_id: str, **fields: object) -> MemoryRecord | None:
        updated = await self.store.update(memory_id, **fields)
        if updated is not None:
            await self._publish(
                EventType.MEMORY_UPDATED,
                memory_id=updated.memory_id,
                memory_type=updated.memory_type.value,
                task_id=updated.task_id,
                session_id=updated.session_id,
            )
        return updated

    async def delete(self, memory_id: str) -> bool:
        deleted = await self.store.delete(memory_id)
        if deleted and self.vector_store is not None and self.settings.vector_enabled:
            await self.vector_store.delete(memory_id)
        if deleted:
            await self._publish(EventType.MEMORY_DELETED, memory_id=memory_id)
        return deleted

    async def list_by(
        self,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
        memory_types: list[MemoryType] | None = None,
        limit: int = 50,
    ) -> list[MemoryRecord]:
        return await self.store.list_by(
            user_id=user_id,
            session_id=session_id,
            task_id=task_id,
            memory_types=memory_types,
            limit=limit,
        )

    async def count(self, *, user_id: str | None = None) -> int:
        return await self.store.count(user_id=user_id)

    # --- Conversation ---
    async def add_conversation_message(
        self, message: ConversationMessage
    ) -> ConversationMessage | None:
        if self.conversation_store is None:
            return None
        try:
            return await self.conversation_store.add(message)
        except Exception as exc:  # noqa: BLE001
            logger.bind(event="memory.conv.add.error", error=type(exc).__name__).warning(
                "Conversation add failed: {}", str(exc)
            )
            return None

    async def list_recent_conversation(
        self,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        limit: int = 20,
    ) -> list[ConversationMessage]:
        if self.conversation_store is None:
            return []
        return await self.conversation_store.list_recent(
            session_id=session_id, user_id=user_id, limit=limit
        )

    def summarize_conversation(self, messages: list[ConversationMessage]) -> str:
        return ConversationSummarizer().summarize(messages)

    # --- Consolidation ---
    async def consolidate(self) -> ConsolidationReport:
        if self.consolidator is None:
            return ConsolidationReport()
        report = await self.consolidator.consolidate()
        await self._publish(
            EventType.MEMORY_CONSOLIDATED,
            payload={
                "deleted_expired": report.deleted_expired,
                "promoted": report.promoted,
                "merged_duplicates": report.merged_duplicates,
            },
        )
        return report

    # --- Health ---
    async def health_check(self) -> MemoryManagerState:
        """Refresh and return the subsystem state snapshot."""
        self._state.record_count = await self._safe_count()
        return self._state

    async def _safe_count(self) -> int:
        try:
            return await self.store.count()
        except Exception:  # noqa: BLE001
            return 0

    # --- Events ---
    async def _publish(
        self,
        event_type: EventType,
        *,
        memory_id: str | None = None,
        memory_type: str | None = None,
        source: str | None = None,
        task_id: str | None = None,
        session_id: str | None = None,
        payload: dict | None = None,
        reason: str | None = None,
    ) -> None:
        """Publish a memory event. **Never** includes memory content."""
        data = {
            "memory_id": memory_id,
            "memory_type": memory_type,
            "source": source,
        }
        if reason is not None:
            data["reason"] = reason
        if payload:
            data.update(payload)
        await self.event_bus.publish(
            Event.create(
                event_type,
                task_id=task_id,
                session_id=session_id,
                payload=data,
                metadata={"memory_id": memory_id, "memory_type": memory_type},
            )
        )


def build_memory_manager(
    settings: MemorySettings,
    event_bus: EventBus,
    *,
    store: MemoryStore | None = None,
    working_client: Redis | None = None,
    vector_store: VectorMemoryStore | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> MemoryManager:
    """Construct a :class:`MemoryManager` from configuration.

    Used by the runtime to wire the manager. When Redis/vector are disabled
    or unavailable, the manager omits them and degrades gracefully.
    """
    durable = store or PostgreSQLStore()
    working = None
    if settings.redis_working_memory and working_client is not None:
        working = RedisWorkingMemoryStore(working_client, settings)

    vector = None
    embed = embedding_provider
    if settings.vector_enabled:
        vector = vector_store or InMemoryVectorStore(embed or HashingEmbeddingProvider())
        if embed is None:
            embed = HashingEmbeddingProvider(settings.vector_dimensions)
    elif embed is None:
        embed = HashingEmbeddingProvider(settings.vector_dimensions)

    conversation_store = ConversationStore()

    return MemoryManager(
        settings=settings,
        event_bus=event_bus,
        store=durable,
        working_memory=working,
        vector_store=vector,
        embedding_provider=embed,
        write_policy=MemoryWritePolicy(settings),
        retriever=MemoryRetriever(
            store=durable,
            vector_store=vector if settings.vector_enabled else None,
            settings=settings,
        ),
        conversation_store=conversation_store,
        consolidator=BasicMemoryConsolidator(durable),
    )
