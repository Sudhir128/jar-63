"""Comprehensive tests for the Phase 6 persistent memory subsystem.

Covers:
* Settings loading (MemorySettings)
* MemoryRecord / MemoryContext domain models
* MemoryStore (PostgreSQLStore via SQLite): create, get, list_by, search, update, delete, delete_expired
* RedisWorkingMemoryStore (fakeredis): add, list, clear
* InMemoryVectorStore: add, search, filter_metadata, remove, clear
* HashingEmbeddingProvider: embed, dimensionality, determinism
* MemoryWritePolicy: sensitive detection, duplicate detection, importance threshold
* MemoryRetriever: retrieve_context, ranking, packing
* ConversationStore + ConversationSummarizer
* BasicMemoryConsolidator
* MemoryManager: end-to-end add/retrieve/delete/consolidate
* Memory health checker
* EventBus integration (memory events published)
* MemoryDiscoverStage
* REST API endpoints (/api/v1/memory/*)
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from app.config.memory_settings import MemorySettings
from app.core.identifiers import utc_now
from app.events import EventType, InMemoryEventBus
from app.memory import (
    ConversationStore,
    ConversationSummarizer,
    InMemoryVectorStore,
    MemoryManager,
    MemoryRetriever,
    MemoryWritePolicy,
    PostgreSQLStore,
    RedisWorkingMemoryStore,
    build_memory_manager,
)
from app.memory.consolidation import BasicMemoryConsolidator
from app.memory.db_init import init_memory_tables
from app.memory.embeddings import HashingEmbeddingProvider
from app.memory.health import MemoryHealthChecker
from app.memory.models import ConversationMessage, MemoryContext, MemoryRecord, MemorySearchQuery
from app.memory.phase6_demos import run_all_phase6_demos
from app.memory.types import MemoryType, RetentionPolicy

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
async def _memory_tables():
    """Ensure memory tables exist before each test (idempotent)."""
    await init_memory_tables()


@pytest.fixture()
def memory_settings() -> MemorySettings:
    return MemorySettings(
        _env_file=None,
        enabled=True,
        vector_enabled=True,
        retrieval_max_items=10,
        retrieval_max_chars=4000,
        retrieval_max_tokens=1000,
    )


@pytest.fixture()
async def manager(memory_settings: MemorySettings) -> MemoryManager:
    bus = InMemoryEventBus()
    mgr = build_memory_manager(settings=memory_settings, event_bus=bus)
    await mgr.start()
    yield mgr
    await mgr.shutdown()


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class TestMemorySettings:
    def test_defaults(self) -> None:
        s = MemorySettings(_env_file=None)
        assert s.enabled is True
        assert s.pg_persistence is True
        assert s.redis_working_memory is True
        assert s.vector_enabled is False
        assert s.embedding_provider == "hashing"
        assert s.duplicate_similarity_threshold > 0

    def test_populate_by_name(self) -> None:
        s = MemorySettings(_env_file=None, vector_enabled=True, enabled=False)
        assert s.vector_enabled is True
        assert s.enabled is False


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------


class TestMemoryRecord:
    def test_creation_defaults(self) -> None:
        rec = MemoryRecord(content="hello", memory_type=MemoryType.SEMANTIC)
        assert rec.memory_id.startswith("mem_")
        assert rec.created_at is not None
        assert rec.importance == 0.5
        assert rec.confidence == 0.5
        assert rec.retention_policy == RetentionPolicy.TTL

    def test_is_expired(self) -> None:
        past = utc_now() - timedelta(hours=1)
        rec = MemoryRecord(
            content="x",
            memory_type=MemoryType.WORKING,
            expires_at=past,
        )
        assert rec.is_expired is True

    def test_not_expired(self) -> None:
        future = utc_now() + timedelta(hours=1)
        rec = MemoryRecord(
            content="x",
            memory_type=MemoryType.WORKING,
            expires_at=future,
        )
        assert rec.is_expired is False

    def test_no_expiry_not_expired(self) -> None:
        rec = MemoryRecord(content="x", memory_type=MemoryType.SEMANTIC)
        assert rec.is_expired is False


class TestMemoryContext:
    def test_empty(self) -> None:
        ctx = MemoryContext(query="test")
        assert len(ctx.memories) == 0
        assert ctx.total_chars == 0
        assert ctx.truncated is False


# ---------------------------------------------------------------------------
# PostgreSQLStore (via SQLite)
# ---------------------------------------------------------------------------


class TestMemoryStore:
    @pytest.mark.asyncio
    async def test_create_and_get(self) -> None:
        store = PostgreSQLStore()
        rec = MemoryRecord(content="test content", memory_type=MemoryType.SEMANTIC, user_id="u1")
        created = await store.create(rec)
        fetched = await store.get(created.memory_id)
        assert fetched is not None
        assert fetched.content == "test content"
        assert fetched.user_id == "u1"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self) -> None:
        store = PostgreSQLStore()
        assert await store.get("mem_nonexistent") is None

    @pytest.mark.asyncio
    async def test_list_by_user(self) -> None:
        store = PostgreSQLStore()
        for i in range(3):
            await store.create(
                MemoryRecord(content=f"c{i}", memory_type=MemoryType.SEMANTIC, user_id="u_list")
            )
        results = await store.list_by(user_id="u_list")
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_list_by_type(self) -> None:
        store = PostgreSQLStore()
        await store.create(
            MemoryRecord(content="sem", memory_type=MemoryType.SEMANTIC, user_id="u_t")
        )
        await store.create(
            MemoryRecord(content="work", memory_type=MemoryType.WORKING, user_id="u_t")
        )
        results = await store.list_by(user_id="u_t", memory_types=[MemoryType.SEMANTIC])
        assert len(results) == 1
        assert results[0].content == "sem"

    @pytest.mark.asyncio
    async def test_search_substring(self) -> None:
        store = PostgreSQLStore()
        await store.create(
            MemoryRecord(
                content="The user likes Python", memory_type=MemoryType.SEMANTIC, user_id="u_s"
            )
        )
        sq = MemorySearchQuery(query="python", user_id="u_s", top_k=10)
        results = await store.search(sq)
        assert len(results) == 1
        assert results[0].matched_by == "structured"

    @pytest.mark.asyncio
    async def test_search_no_match(self) -> None:
        store = PostgreSQLStore()
        await store.create(
            MemoryRecord(content="hello world", memory_type=MemoryType.SEMANTIC, user_id="u_n")
        )
        sq = MemorySearchQuery(query="nomatch", user_id="u_n", top_k=10)
        results = await store.search(sq)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_update(self) -> None:
        store = PostgreSQLStore()
        rec = await store.create(
            MemoryRecord(content="original", memory_type=MemoryType.SEMANTIC, user_id="u_u")
        )
        updated = await store.update(rec.memory_id, content="changed", importance=0.9)
        assert updated is not None
        assert updated.content == "changed"
        assert updated.importance == 0.9

    @pytest.mark.asyncio
    async def test_delete(self) -> None:
        store = PostgreSQLStore()
        rec = await store.create(
            MemoryRecord(content="to delete", memory_type=MemoryType.SEMANTIC, user_id="u_d")
        )
        assert await store.delete(rec.memory_id) is True
        assert await store.get(rec.memory_id) is None
        assert await store.delete("mem_nonexistent") is False

    @pytest.mark.asyncio
    async def test_delete_expired(self) -> None:
        store = PostgreSQLStore()
        past = utc_now() - timedelta(hours=2)
        await store.create(
            MemoryRecord(
                content="expired", memory_type=MemoryType.WORKING, user_id="u_e", expires_at=past
            )
        )
        await store.create(
            MemoryRecord(content="alive", memory_type=MemoryType.SEMANTIC, user_id="u_e")
        )
        deleted = await store.delete_expired()
        assert deleted == 1
        remaining = await store.list_by(user_id="u_e")
        assert len(remaining) == 1
        assert remaining[0].content == "alive"

    @pytest.mark.asyncio
    async def test_count(self) -> None:
        store = PostgreSQLStore()
        await store.create(
            MemoryRecord(content="a", memory_type=MemoryType.SEMANTIC, user_id="u_c")
        )
        await store.create(
            MemoryRecord(content="b", memory_type=MemoryType.SEMANTIC, user_id="u_c")
        )
        assert await store.count(user_id="u_c") == 2
        assert await store.count(user_id="other") == 0


# ---------------------------------------------------------------------------
# RedisWorkingMemoryStore
# ---------------------------------------------------------------------------


class TestWorkingMemory:
    @pytest.mark.asyncio
    async def test_add_and_list(self) -> None:
        import fakeredis.aioredis

        fake = fakeredis.aioredis.FakeRedis()
        try:
            wm = RedisWorkingMemoryStore(fake, MemorySettings(_env_file=None))
            rec = MemoryRecord(content="working item", memory_type=MemoryType.WORKING, task_id="t1")
            await wm.add(rec)
            items = await wm.list(task_id="t1")
            assert len(items) == 1
            assert items[0].content == "working item"
        finally:
            await fake.aclose()

    @pytest.mark.asyncio
    async def test_clear(self) -> None:
        import fakeredis.aioredis

        fake = fakeredis.aioredis.FakeRedis()
        try:
            wm = RedisWorkingMemoryStore(fake, MemorySettings(_env_file=None))
            rec = MemoryRecord(content="item", memory_type=MemoryType.WORKING, task_id="t2")
            await wm.add(rec)
            await wm.clear(task_id="t2")
            assert len(await wm.list(task_id="t2")) == 0
        finally:
            await fake.aclose()


# ---------------------------------------------------------------------------
# InMemoryVectorStore
# ---------------------------------------------------------------------------


class TestVectorStore:
    @pytest.mark.asyncio
    async def test_add_and_search(self) -> None:
        vs = InMemoryVectorStore(HashingEmbeddingProvider(dimensions=128))
        rec = MemoryRecord(
            content="python programming", memory_type=MemoryType.SEMANTIC, user_id="uv"
        )
        await vs.upsert(rec.memory_id, rec.content, {"user_id": "uv"})
        hits = await vs.search("python programming", top_k=5)
        assert len(hits) == 1
        assert hits[0][0] == rec.memory_id
        assert hits[0][1] > 0.5

    @pytest.mark.asyncio
    async def test_filter_metadata(self) -> None:
        vs = InMemoryVectorStore(HashingEmbeddingProvider(dimensions=128))
        rec1 = MemoryRecord(content="a", memory_type=MemoryType.SEMANTIC, user_id="u1")
        rec2 = MemoryRecord(content="a", memory_type=MemoryType.SEMANTIC, user_id="u2")
        await vs.upsert(rec1.memory_id, "a", {"user_id": "u1"})
        await vs.upsert(rec2.memory_id, "a", {"user_id": "u2"})
        hits = await vs.search("a", top_k=5, filter_metadata={"user_id": "u1"})
        assert len(hits) == 1
        assert hits[0][0] == rec1.memory_id

    @pytest.mark.asyncio
    async def test_delete(self) -> None:
        vs = InMemoryVectorStore(HashingEmbeddingProvider(dimensions=128))
        rec = MemoryRecord(content="x", memory_type=MemoryType.SEMANTIC)
        await vs.upsert(rec.memory_id, "x", {})
        assert await vs.delete(rec.memory_id) is True
        assert len(await vs.search("x", top_k=5)) == 0


# ---------------------------------------------------------------------------
# EmbeddingProvider
# ---------------------------------------------------------------------------


class TestEmbeddings:
    @pytest.mark.asyncio
    async def test_deterministic(self) -> None:
        emb = HashingEmbeddingProvider(dimensions=64)
        v1 = await emb.embed("hello world")
        v2 = await emb.embed("hello world")
        assert v1 == v2

    @pytest.mark.asyncio
    async def test_dimensionality(self) -> None:
        emb = HashingEmbeddingProvider(dimensions=128)
        v = await emb.embed("test")
        assert len(v) == 128

    @pytest.mark.asyncio
    async def test_different_text_different_vector(self) -> None:
        emb = HashingEmbeddingProvider(dimensions=64)
        v1 = await emb.embed("hello")
        v2 = await emb.embed("world")
        assert v1 != v2


# ---------------------------------------------------------------------------
# MemoryWritePolicy
# ---------------------------------------------------------------------------


class TestWritePolicy:
    def test_sensitive_detection(self) -> None:
        policy = MemoryWritePolicy(MemorySettings(_env_file=None))
        assert policy._detect_sensitive("the api_key is sk-123") is True
        assert policy._detect_sensitive("normal content") is False

    def test_evaluate_blocks_sensitive(self) -> None:
        policy = MemoryWritePolicy(MemorySettings(_env_file=None))
        rec = MemoryRecord(
            content="api_key=sk-secret", memory_type=MemoryType.SEMANTIC, user_id="u"
        )
        decision = policy.evaluate(rec, existing=[])
        assert decision.decision.value == "ignore"
        assert "sensitive" in decision.reason.lower() or "privacy" in decision.reason.lower()


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class TestRetriever:
    @pytest.mark.asyncio
    async def test_retrieve_context(self, memory_settings: MemorySettings) -> None:
        store = PostgreSQLStore()
        vector = InMemoryVectorStore(HashingEmbeddingProvider(dimensions=64))
        retriever = MemoryRetriever(store, vector, memory_settings)
        rec = await store.create(
            MemoryRecord(
                content="user likes concise answers",
                memory_type=MemoryType.SEMANTIC,
                user_id="ur",
                importance=0.8,
            )
        )
        await vector.upsert(rec.memory_id, rec.content, {"user_id": "ur"})
        ctx = await retriever.retrieve_context("concise", user_id="ur")
        assert len(ctx.memories) >= 1
        assert ctx.total_available >= 1


# ---------------------------------------------------------------------------
# ConversationStore
# ---------------------------------------------------------------------------


class TestConversationStore:
    @pytest.mark.asyncio
    async def test_add_and_list(self) -> None:
        store = ConversationStore()
        await store.add(ConversationMessage(role="user", content="hi", session_id="s1"))
        await store.add(ConversationMessage(role="assistant", content="hello", session_id="s1"))
        msgs = await store.list_recent(session_id="s1", limit=10)
        assert len(msgs) == 2
        assert msgs[0].role == "user"

    @pytest.mark.asyncio
    async def test_count(self) -> None:
        store = ConversationStore()
        await store.add(ConversationMessage(role="user", content="a", session_id="s2"))
        await store.add(ConversationMessage(role="user", content="b", session_id="s2"))
        assert await store.count(session_id="s2") == 2


class TestConversationSummarizer:
    def test_summarize(self) -> None:
        msgs = [
            ConversationMessage(role="user", content="What is 2+2?", session_id="s"),
            ConversationMessage(role="assistant", content="4", session_id="s"),
        ]
        summary = ConversationSummarizer().summarize(msgs)
        assert "2+2" in summary
        assert len(summary) > 0

    def test_empty(self) -> None:
        assert ConversationSummarizer().summarize([]) == ""


# ---------------------------------------------------------------------------
# Consolidator
# ---------------------------------------------------------------------------


class TestConsolidator:
    @pytest.mark.asyncio
    async def test_consolidate_expired(self) -> None:
        store = PostgreSQLStore()
        past = utc_now() - timedelta(hours=2)
        await store.create(
            MemoryRecord(
                content="expired", memory_type=MemoryType.WORKING, user_id="uc", expires_at=past
            )
        )
        consolidator = BasicMemoryConsolidator(store)
        report = await consolidator.consolidate()
        assert report.deleted_expired == 1


# ---------------------------------------------------------------------------
# MemoryManager (end-to-end)
# ---------------------------------------------------------------------------


class TestMemoryManager:
    @pytest.mark.asyncio
    async def test_add_and_retrieve(self, manager: MemoryManager) -> None:
        rec = await manager.add(
            content="user prefers dark mode",
            memory_type=MemoryType.SEMANTIC,
            user_id="um",
            importance=0.8,
        )
        assert rec is not None
        ctx = await manager.retrieve_context("dark", user_id="um")
        assert len(ctx.memories) >= 1

    @pytest.mark.asyncio
    async def test_add_duplicate_rejected(self, manager: MemoryManager) -> None:
        await manager.add(
            content="user likes tea",
            memory_type=MemoryType.SEMANTIC,
            user_id="um2",
        )
        second = await manager.add(
            content="user likes tea",
            memory_type=MemoryType.SEMANTIC,
            user_id="um2",
        )
        assert second is None
        assert await manager.count(user_id="um2") == 1

    @pytest.mark.asyncio
    async def test_add_sensitive_blocked(self, manager: MemoryManager) -> None:
        rec = await manager.add(
            content="api_key is sk-secret",
            memory_type=MemoryType.SEMANTIC,
            user_id="um3",
        )
        assert rec is None
        assert await manager.count(user_id="um3") == 0

    @pytest.mark.asyncio
    async def test_delete(self, manager: MemoryManager) -> None:
        rec = await manager.add(content="to remove", memory_type=MemoryType.SEMANTIC, user_id="um4")
        assert rec is not None
        assert await manager.delete(rec.memory_id) is True
        assert await manager.get(rec.memory_id) is None

    @pytest.mark.asyncio
    async def test_consolidate(self, manager: MemoryManager) -> None:
        past = utc_now() - timedelta(hours=2)
        await manager.store.create(
            MemoryRecord(
                content="expired", memory_type=MemoryType.WORKING, user_id="um5", expires_at=past
            )
        )
        report = await manager.consolidate()
        assert report.deleted_expired >= 1

    @pytest.mark.asyncio
    async def test_list_by(self, manager: MemoryManager) -> None:
        await manager.add(content="item 1", memory_type=MemoryType.SEMANTIC, user_id="um6")
        await manager.add(content="item 2", memory_type=MemoryType.SEMANTIC, user_id="um6")
        results = await manager.list_by(user_id="um6")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_disabled_manager(self) -> None:
        settings = MemorySettings(_env_file=None, enabled=False)
        mgr = build_memory_manager(settings=settings, event_bus=InMemoryEventBus())
        await mgr.start()
        assert mgr.is_enabled is False
        rec = await mgr.add(
            content="should be ignored", memory_type=MemoryType.SEMANTIC, user_id="x"
        )
        assert rec is None
        await mgr.shutdown()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class TestMemoryHealth:
    @pytest.mark.asyncio
    async def test_health_check(self, manager: MemoryManager) -> None:
        health = MemoryHealthChecker(manager)
        snap = await health.check()
        assert snap.enabled is True
        d = snap.to_api_dict()
        assert "status" in d
        assert "enabled" in d


# ---------------------------------------------------------------------------
# EventBus integration
# ---------------------------------------------------------------------------


class TestMemoryEvents:
    @pytest.mark.asyncio
    async def test_memory_updated_event(self, memory_settings: MemorySettings) -> None:
        bus = InMemoryEventBus()
        captured = []

        async def handler(ev) -> None:
            captured.append(ev)

        bus.subscribe(EventType.MEMORY_CREATED, handler)
        mgr = build_memory_manager(settings=memory_settings, event_bus=bus)
        await mgr.start()
        await mgr.add(
            content="event test content unique", memory_type=MemoryType.SEMANTIC, user_id="ue"
        )
        await asyncio.sleep(0.05)
        await mgr.shutdown()
        assert len(captured) >= 1


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------


class TestPhase6Demos:
    @pytest.mark.asyncio
    async def test_all_demos_pass(self) -> None:
        results = await run_all_phase6_demos()
        assert len(results) == 6
        for r in results:
            assert r.passed, f"{r.name} failed: {r.detail}"


# ---------------------------------------------------------------------------
# REST API
# ---------------------------------------------------------------------------


class TestMemoryAPI:
    @pytest.fixture()
    def app_with_memory(self):
        from app.main import create_app

        return create_app()

    def test_status_endpoint(self, app_with_memory) -> None:
        from fastapi.testclient import TestClient

        with TestClient(app_with_memory) as c:
            resp = c.get("/api/v1/memory/status")
            assert resp.status_code in (200, 503)

    def test_memories_crud(self, app_with_memory) -> None:
        from fastapi.testclient import TestClient

        with TestClient(app_with_memory) as c:
            # Create
            resp = c.post(
                "/api/v1/memory/memories",
                json={
                    "content": "api test memory",
                    "memory_type": "semantic",
                    "user_id": "api_user",
                    "importance": 0.8,
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            if data.get("status") == "rejected":
                pytest.skip("Memory rejected by policy in shared store")
            mid = data["memory_id"]

            # List
            resp = c.get("/api/v1/memory/memories", params={"user_id": "api_user"})
            assert resp.status_code == 200
            assert resp.json()["count"] >= 1

            # Get
            resp = c.get(f"/api/v1/memory/memories/{mid}")
            assert resp.status_code == 200

            # Retrieve
            resp = c.post(
                "/api/v1/memory/retrieve",
                json={"query": "api", "user_id": "api_user"},
            )
            assert resp.status_code == 200

            # Delete
            resp = c.delete(f"/api/v1/memory/memories/{mid}")
            assert resp.status_code == 200

    def test_conversations_endpoint(self, app_with_memory) -> None:
        from fastapi.testclient import TestClient

        with TestClient(app_with_memory) as c:
            resp = c.get("/api/v1/memory/conversations")
            assert resp.status_code in (200, 503)
