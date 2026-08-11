"""Phase 6 demonstration workflows: persistent memory subsystem.

Six deterministic demos that exercise the memory subsystem end-to-end
**without requiring an LLM or external services**. They use an in-process
SQLite durable store and (optionally) fakeredis for working memory.

* Demo 1 — Store + retrieve a semantic memory (round-trip)
* Demo 2 — Duplicate detection (identical content → update/ignore)
* Demo 3 — Privacy guardrail (sensitive pattern → write rejected)
* Demo 4 — Working memory round-trip (Redis via fakeredis)
* Demo 5 — Conversation store + summarizer
* Demo 6 — Consolidation (expiry + dedup)

Each demo returns a :class:`Phase6DemoResult` with ``passed`` and ``detail``.
Results are not faked — the demos actually run the code paths.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.events import EventBus, InMemoryEventBus
from app.memory import (
    ConversationStore,
    ConversationSummarizer,
    MemoryManager,
    RedisWorkingMemoryStore,
    build_memory_manager,
)
from app.memory.models import ConversationMessage, MemoryRecord
from app.memory.types import MemoryType

__all__ = [
    "Phase6DemoResult",
    "run_store_retrieve_demo",
    "run_duplicate_detection_demo",
    "run_privacy_guardrail_demo",
    "run_working_memory_demo",
    "run_conversation_demo",
    "run_consolidation_demo",
    "run_all_phase6_demos",
]


def _uid() -> str:
    """Short unique suffix so demos don't collide across runs in a shared DB."""
    return uuid.uuid4().hex[:8]


@dataclass
class Phase6DemoResult:
    name: str
    passed: bool
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)


def _make_manager(
    *,
    event_bus: EventBus | None = None,
    with_vector: bool = True,
    with_working: bool = False,
) -> MemoryManager:
    """Build a memory manager backed by the shared test SQLite engine."""
    from app.config.memory_settings import MemorySettings

    bus = event_bus or InMemoryEventBus()
    settings = MemorySettings(
        _env_file=None,
        enabled=True,
        pg_persistence=True,
        vector_enabled=with_vector,
        redis_working_memory=with_working,
        embedding="hashing",
    )
    working_client = None
    if with_working:
        import fakeredis.aioredis

        working_client = fakeredis.aioredis.FakeRedis()
    return build_memory_manager(
        settings=settings,
        event_bus=bus,
        working_client=working_client,
    )


async def run_store_retrieve_demo() -> Phase6DemoResult:
    """Demo 1: store a semantic memory and retrieve it by query."""
    from app.memory.db_init import init_memory_tables

    await init_memory_tables()
    manager = _make_manager()
    await manager.start()
    try:
        uid = f"user-{_uid()}"
        rec = await manager.add(
            content="The user prefers concise answers without preamble.",
            memory_type=MemoryType.SEMANTIC,
            user_id=uid,
            importance=0.8,
            confidence=0.9,
        )
        if rec is None:
            return Phase6DemoResult("store_retrieve", False, "memory add returned None")
        ctx = await manager.retrieve_context("concise", user_id=uid)
        if len(ctx.memories) == 0:
            return Phase6DemoResult("store_retrieve", False, "no memories retrieved")
        found = ctx.memories[0]
        if "concise" not in found.content:
            return Phase6DemoResult("store_retrieve", False, "wrong memory retrieved")
        return Phase6DemoResult(
            "store_retrieve",
            True,
            f"stored+retrieved memory_id={rec.memory_id}",
            data={"memory_id": rec.memory_id, "retrieved_count": len(ctx.memories)},
        )
    finally:
        await manager.shutdown()


async def run_duplicate_detection_demo() -> Phase6DemoResult:
    """Demo 2: adding near-duplicate content does not create a second record."""
    from app.memory.db_init import init_memory_tables

    await init_memory_tables()
    manager = _make_manager()
    await manager.start()
    try:
        uid = f"user-{_uid()}"
        await manager.add(
            content="User likes Python 3.13.",
            memory_type=MemoryType.SEMANTIC,
            user_id=uid,
            importance=0.7,
        )
        second = await manager.add(
            content="User likes Python 3.13.",
            memory_type=MemoryType.SEMANTIC,
            user_id=uid,
            importance=0.7,
        )
        count = await manager.count(user_id=uid)
        if count != 1:
            return Phase6DemoResult(
                "duplicate_detection",
                False,
                f"expected 1 record, got {count}",
            )
        if second is not None:
            return Phase6DemoResult(
                "duplicate_detection",
                False,
                "duplicate was stored instead of rejected",
            )
        return Phase6DemoResult(
            "duplicate_detection",
            True,
            "duplicate correctly rejected",
            data={"count": count},
        )
    finally:
        await manager.shutdown()


async def run_privacy_guardrail_demo() -> Phase6DemoResult:
    """Demo 3: content matching a sensitive pattern is rejected."""
    from app.memory.db_init import init_memory_tables

    await init_memory_tables()
    manager = _make_manager()
    await manager.start()
    try:
        uid = f"user-{_uid()}"
        rec = await manager.add(
            content="The api_key for prod is sk-1234567890abcdef.",
            memory_type=MemoryType.SEMANTIC,
            user_id=uid,
        )
        if rec is not None:
            return Phase6DemoResult(
                "privacy_guardrail",
                False,
                "sensitive content was stored (should be rejected)",
            )
        count = await manager.count(user_id=uid)
        if count != 0:
            return Phase6DemoResult(
                "privacy_guardrail",
                False,
                f"sensitive content found in store ({count})",
            )
        return Phase6DemoResult(
            "privacy_guardrail",
            True,
            "sensitive content blocked by policy",
        )
    finally:
        await manager.shutdown()


async def run_working_memory_demo() -> Phase6DemoResult:
    """Demo 4: working memory round-trip via fakeredis."""
    import fakeredis.aioredis

    from app.config.memory_settings import MemorySettings

    fake = fakeredis.aioredis.FakeRedis()
    settings = MemorySettings(_env_file=None, enabled=True, redis_working_memory=True)
    wm = RedisWorkingMemoryStore(fake, settings)
    record = MemoryRecord(
        content="intermediate result: 42",
        memory_type=MemoryType.WORKING,
        session_id="sess-1",
        task_id="task-1",
    )
    await wm.add(record)
    items = await wm.list(task_id="task-1")
    await wm.clear(task_id="task-1")
    after_clear = await wm.list(task_id="task-1")
    if len(items) != 1:
        return Phase6DemoResult("working_memory", False, f"expected 1 item, got {len(items)}")
    if items[0].content != "intermediate result: 42":
        return Phase6DemoResult("working_memory", False, "wrong content retrieved")
    if len(after_clear) != 0:
        return Phase6DemoResult("working_memory", False, "clear did not remove items")
    await fake.aclose()
    return Phase6DemoResult("working_memory", True, "add+list+clear OK")


async def run_conversation_demo() -> Phase6DemoResult:
    """Demo 5: conversation store + deterministic summarizer."""
    from app.memory.db_init import init_memory_tables

    await init_memory_tables()
    store = ConversationStore()
    sid = f"s1_demo_{_uid()}"
    messages = [
        ConversationMessage(role="user", content="What is 2+2?", session_id=sid),
        ConversationMessage(role="assistant", content="4", session_id=sid),
        ConversationMessage(role="user", content="What is the weather?", session_id=sid),
    ]
    for m in messages:
        await store.add(m)
    recent = await store.list_recent(session_id=sid, limit=10)
    summary = ConversationSummarizer().summarize(recent)
    count = await store.count(session_id=sid)
    if count != 3:
        return Phase6DemoResult("conversation", False, f"expected 3 messages, got {count}")
    if len(recent) != 3:
        return Phase6DemoResult("conversation", False, f"expected 3 recent, got {len(recent)}")
    if not summary or "2+2" not in summary:
        return Phase6DemoResult("conversation", False, "summary missing user turn")
    return Phase6DemoResult(
        "conversation", True, f"stored+summarized {count} messages", data={"summary": summary}
    )


async def run_consolidation_demo() -> Phase6DemoResult:
    """Demo 6: consolidation merges exact duplicates and deletes expired."""
    from datetime import timedelta

    from app.core.identifiers import utc_now
    from app.memory.db_init import init_memory_tables

    await init_memory_tables()
    manager = _make_manager(with_vector=False)
    await manager.start()
    try:
        uid = f"u-{_uid()}"
        # Two exact duplicates (bypass policy by creating directly via store).
        past = utc_now() - timedelta(days=1)
        expired = utc_now() - timedelta(hours=2)
        rec1 = MemoryRecord(
            content="dup content",
            memory_type=MemoryType.SEMANTIC,
            user_id=uid,
            created_at=past,
        )
        rec2 = MemoryRecord(
            content="dup content",
            memory_type=MemoryType.SEMANTIC,
            user_id=uid,
            created_at=past,
        )
        rec3 = MemoryRecord(
            content="expired content",
            memory_type=MemoryType.WORKING,
            user_id=uid,
            expires_at=expired,
        )
        await manager.store.create(rec1)
        await manager.store.create(rec2)
        await manager.store.create(rec3)
        before = await manager.count()
        report = await manager.consolidate()
        after = await manager.count()
        if report.merged_duplicates < 1:
            return Phase6DemoResult(
                "consolidation",
                False,
                f"expected ≥1 merge, got {report.merged_duplicates}",
            )
        if report.deleted_expired < 1:
            return Phase6DemoResult(
                "consolidation",
                False,
                f"expected ≥1 expired deletion, got {report.deleted_expired}",
            )
        if after >= before:
            return Phase6DemoResult(
                "consolidation",
                False,
                f"count did not decrease: before={before} after={after}",
            )
        return Phase6DemoResult(
            "consolidation",
            True,
            f"merged={report.merged_duplicates} expired={report.deleted_expired}",
            data={"before": before, "after": after},
        )
    finally:
        await manager.shutdown()


async def run_all_phase6_demos() -> list[Phase6DemoResult]:
    """Run all six Phase 6 demos and return their results."""
    results = [
        await run_store_retrieve_demo(),
        await run_duplicate_detection_demo(),
        await run_privacy_guardrail_demo(),
        await run_working_memory_demo(),
        await run_conversation_demo(),
        await run_consolidation_demo(),
    ]
    return results


def run_all_phase6_demos_sync() -> list[Phase6DemoResult]:
    """Synchronous wrapper for running all demos."""
    return asyncio.run(run_all_phase6_demos())
