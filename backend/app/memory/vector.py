"""Vector memory store abstraction and in-process implementation (Phase 6).

The :class:`VectorMemoryStore` ABC defines the vector-index contract:
embed, upsert, similarity search, delete. The default
:class:`InMemoryVectorStore` is a deterministic, in-process implementation
suitable for testing and small-scale local operation. A future phase can
swap in a pgvector-backed store without changing callers.

Cosine similarity is used throughout. Vectors are L2-normalized at embed
time, so cosine similarity reduces to a dot product.
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field

from app.core.logging import get_logger
from app.memory.embeddings import EmbeddingProvider, HashingEmbeddingProvider

logger = get_logger("memory.vector")

__all__ = ["VectorRecord", "VectorMemoryStore", "InMemoryVectorStore"]


@dataclass(frozen=True)
class VectorRecord:
    """A stored vector keyed by memory_id."""

    memory_id: str
    vector: list[float]
    metadata: dict[str, object] = field(default_factory=dict)


class VectorMemoryStore(abc.ABC):
    """Vector index contract for semantic memory retrieval."""

    @abc.abstractmethod
    async def upsert(self, memory_id: str, text: str, metadata: dict[str, object]) -> None:
        """Embed ``text`` and store/update the vector for ``memory_id``."""

    @abc.abstractmethod
    async def upsert_vector(
        self, memory_id: str, vector: list[float], metadata: dict[str, object]
    ) -> None:
        """Store a pre-computed vector for ``memory_id``."""

    @abc.abstractmethod
    async def search(
        self,
        query_text: str,
        *,
        top_k: int = 10,
        filter_metadata: dict[str, object] | None = None,
    ) -> list[tuple[str, float]]:
        """Return ``(memory_id, score)`` pairs ranked by cosine similarity."""

    @abc.abstractmethod
    async def search_vector(
        self,
        query_vector: list[float],
        *,
        top_k: int = 10,
        filter_metadata: dict[str, object] | None = None,
    ) -> list[tuple[str, float]]:
        """Search with a pre-computed query vector."""

    @abc.abstractmethod
    async def delete(self, memory_id: str) -> bool:
        """Remove the vector for ``memory_id``."""

    @abc.abstractmethod
    async def count(self) -> int:
        """Return the number of stored vectors."""


class InMemoryVectorStore(VectorMemoryStore):
    """Deterministic, in-process vector store.

    Stores L2-normalized vectors and ranks by cosine similarity (dot product
    of normalized vectors). Metadata filters perform exact equality matches
    on provided keys. No persistence: state is lost on restart.
    """

    def __init__(self, embedding_provider: EmbeddingProvider | None = None) -> None:
        self._provider = embedding_provider or HashingEmbeddingProvider()
        self._records: dict[str, VectorRecord] = {}

    async def upsert(self, memory_id: str, text: str, metadata: dict[str, object]) -> None:
        vec = await self._provider.embed(text)
        await self.upsert_vector(memory_id, vec, metadata)

    async def upsert_vector(
        self, memory_id: str, vector: list[float], metadata: dict[str, object]
    ) -> None:
        self._records[memory_id] = VectorRecord(
            memory_id=memory_id, vector=list(vector), metadata=dict(metadata)
        )

    async def search(
        self,
        query_text: str,
        *,
        top_k: int = 10,
        filter_metadata: dict[str, object] | None = None,
    ) -> list[tuple[str, float]]:
        qv = await self._provider.embed(query_text)
        return await self.search_vector(qv, top_k=top_k, filter_metadata=filter_metadata)

    async def search_vector(
        self,
        query_vector: list[float],
        *,
        top_k: int = 10,
        filter_metadata: dict[str, object] | None = None,
    ) -> list[tuple[str, float]]:
        scores: list[tuple[str, float]] = []
        for mid, rec in self._records.items():
            if not _matches_filter(rec.metadata, filter_metadata):
                continue
            score = _cosine(query_vector, rec.vector)
            if score > 0.0:
                scores.append((mid, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    async def delete(self, memory_id: str) -> bool:
        return self._records.pop(memory_id, None) is not None

    async def count(self) -> int:
        return len(self._records)


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return dot / (na * nb)


def _matches_filter(metadata: dict[str, object], filt: dict[str, object] | None) -> bool:
    if not filt:
        return True
    return all(metadata.get(k) == v for k, v in filt.items())
