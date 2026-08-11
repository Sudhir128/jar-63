"""Memory retriever: query → filter → rank → top-k → bounded context (Phase 6).

The :class:`MemoryRetriever` orchestrates retrieval across the structured
(:class:`MemoryStore`) and semantic (:class:`VectorMemoryStore`) backends,
merges results, ranks them, and packs a bounded :class:`MemoryContext`
respecting item-count, character, and token budgets.

Ranking is deterministic and combines:
* relevance score (from structured/vector match)
* importance
* recency decay
* confidence

Memories are treated as **untrusted contextual information**: they never
override policy and are clearly labeled in the prompt section.
"""

from __future__ import annotations

from datetime import UTC
from typing import TYPE_CHECKING

from app.config import MemorySettings
from app.core.identifiers import utc_now
from app.core.logging import get_logger
from app.memory.models import MemoryContext, MemoryRecord, MemorySearchQuery, MemorySearchResult

if TYPE_CHECKING:
    from app.memory.store import MemoryStore
    from app.memory.vector import VectorMemoryStore

logger = get_logger("memory.retriever")

__all__ = ["MemoryRetriever"]


class MemoryRetriever:
    """Hybrid structured + semantic retriever producing a bounded context."""

    def __init__(
        self,
        store: MemoryStore,
        vector_store: VectorMemoryStore | None = None,
        settings: MemorySettings | None = None,
    ) -> None:
        self._store = store
        self._vector = vector_store
        self._settings = settings or MemorySettings()

    async def retrieve_context(
        self,
        query: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
        agent_id: str | None = None,
        memory_types: list | None = None,
    ) -> MemoryContext:
        """Retrieve and pack a bounded :class:`MemoryContext` for a query."""
        sq = MemorySearchQuery(
            query=query,
            user_id=user_id,
            session_id=session_id,
            task_id=task_id,
            agent_id=agent_id,
            memory_types=list(memory_types or []),
            top_k=self._settings.retrieval_max_items,
            use_vector=self._settings.vector_enabled,
        )
        results = await self._gather(sq)
        ranked = self._rank(results)
        total_available = len(ranked)
        packed = self._pack(
            ranked, query=query, user_id=user_id, session_id=session_id, task_id=task_id
        )
        packed = packed.model_copy(update={"total_available": total_available})
        return packed

    async def search(self, query: MemorySearchQuery) -> list[MemorySearchResult]:
        """Run a raw search and return ranked results (unpacked)."""
        results = await self._gather(query)
        return self._rank(results)

    async def _gather(self, query: MemorySearchQuery) -> list[MemorySearchResult]:
        """Collect candidates from structured and (optionally) vector backends."""
        structured = await self._store.search(query)

        vector_hits: list[MemorySearchResult] = []
        if self._vector is not None and query.use_vector and query.query:
            filt = _metadata_filter(query)
            hits = await self._vector.search(query.query, top_k=query.top_k, filter_metadata=filt)
            # Hydrate vector hits into full records.
            for mid, score in hits:
                rec = await self._store.get(mid)
                if rec is None or rec.is_expired:
                    continue
                if query.memory_types and rec.memory_type not in query.memory_types:
                    continue
                vector_hits.append(MemorySearchResult(memory=rec, score=score, matched_by="vector"))

        return _merge(structured, vector_hits)

    def _rank(self, results: list[MemorySearchResult]) -> list[MemorySearchResult]:
        """Deterministic ranking combining relevance, importance, recency, confidence."""
        now = utc_now()
        for r in results:
            recency = _recency_decay(r.memory.created_at, now)
            r_score = r.score if r.matched_by == "vector" else r.score * 0.7
            combined = (
                0.45 * r_score
                + 0.25 * r.memory.importance
                + 0.15 * recency
                + 0.15 * r.memory.confidence
            )
            r = r.model_copy(update={"score": combined})  # noqa: PLW2901
        return sorted(results, key=lambda r: r.score, reverse=True)

    def _pack(
        self,
        ranked: list[MemorySearchResult],
        *,
        query: str,
        user_id: str | None,
        session_id: str | None,
        task_id: str | None,
    ) -> MemoryContext:
        """Pack ranked results into a budget-bounded context."""
        max_items = self._settings.retrieval_max_items
        max_chars = self._settings.retrieval_max_chars
        max_tokens = self._settings.retrieval_max_tokens
        chars_per_token = max(self._settings.chars_per_token_approx, 1)

        selected: list[MemoryRecord] = []
        total_chars = 0
        total_tokens = 0
        truncated = False

        for r in ranked:
            if len(selected) >= max_items:
                truncated = True
                break
            content = r.memory.content or ""
            chars = len(content)
            tokens = max(1, chars // chars_per_token)
            if total_chars + chars > max_chars or total_tokens + tokens > max_tokens:
                truncated = True
                break
            selected.append(r.memory)
            total_chars += chars
            total_tokens += tokens

        return MemoryContext(
            memories=selected,
            total_available=len(ranked),
            total_chars=total_chars,
            total_tokens_approx=total_tokens,
            truncated=truncated,
            query=query,
            user_id=user_id,
            session_id=session_id,
            task_id=task_id,
        )


def _metadata_filter(query: MemorySearchQuery) -> dict[str, object]:
    filt: dict[str, object] = {}
    if query.user_id is not None:
        filt["user_id"] = query.user_id
    if query.session_id is not None:
        filt["session_id"] = query.session_id
    if query.task_id is not None:
        filt["task_id"] = query.task_id
    if query.agent_id is not None:
        filt["agent_id"] = query.agent_id
    return filt


def _merge(
    structured: list[MemorySearchResult], vector_hits: list[MemorySearchResult]
) -> list[MemorySearchResult]:
    """Merge results, preferring the higher score for duplicate IDs."""
    by_id: dict[str, MemorySearchResult] = {}
    for r in structured:
        by_id[r.memory.memory_id] = r
    for r in vector_hits:
        existing = by_id.get(r.memory.memory_id)
        if existing is None or r.score > existing.score:
            by_id[r.memory.memory_id] = r
    return list(by_id.values())


def _recency_decay(created_at, now) -> float:
    """Exponential recency decay in [0, 1] over a 30-day half-life."""
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    delta = now - created_at
    days = max(delta.total_seconds() / 86400.0, 0.0)
    half_life_days = 30.0
    import math

    return math.pow(0.5, days / half_life_days)
