"""Embedding provider abstraction (Phase 6).

The :class:`EmbeddingProvider` ABC decouples the memory subsystem from any
specific embedding model. The default :class:`HashingEmbeddingProvider` is a
**deterministic, local** provider — it produces fixed-size float vectors from
a hash of the text. It is *not* a semantic embedding model; it exists so the
memory subsystem is fully testable and runnable without a GPU or a cloud
API. A future phase can swap in a local Ollama embedding model (e.g.
``nomic-embed-text``) or a cloud provider without changing callers.

Security: embedding inputs are treated as untrusted text. The provider never
logs raw content (callers handle logging), and no network is used by the
default provider.
"""

from __future__ import annotations

import abc
import hashlib
import math

from app.core.logging import get_logger

logger = get_logger("memory.embedding")

__all__ = ["EmbeddingProvider", "HashingEmbeddingProvider"]


class EmbeddingProvider(abc.ABC):
    """Produces a fixed-size float vector for a piece of text."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Provider identifier (e.g. ``hashing``, ``ollama``)."""

    @property
    @abc.abstractmethod
    def dimensions(self) -> int:
        """The dimensionality of every produced vector."""

    @abc.abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Return the embedding vector for ``text``."""

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts (default: sequential)."""
        return [await self.embed(t) for t in texts]


class HashingEmbeddingProvider(EmbeddingProvider):
    """Deterministic, local, non-semantic embedding provider.

    Uses SHA-256 of the normalized text, folded into ``dimensions`` floats
    in ``[-1, 1]``. The same text always yields the same vector, so cosine
    similarity is deterministic and exact-string matches score ~1.0.

    This is explicitly **not** a semantic model: it cannot capture meaning.
    It exists for offline/CI correctness and as a placeholder until a real
    local embedding model is configured.
    """

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self._dimensions = dimensions

    @property
    def name(self) -> str:
        return "hashing"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, text: str) -> list[float]:
        normalized = _normalize(text)
        if not normalized:
            return [0.0] * self._dimensions
        digest = hashlib.sha256(normalized.encode("utf-8")).digest()
        # Expand the digest deterministically to fill ``dimensions``.
        values: list[float] = []
        seed = digest
        while len(values) < self._dimensions:
            values.extend(b / 255.0 for b in seed)
            seed = hashlib.sha256(seed + digest).digest()
        # Map [0, 1] → [-1, 1], then L2-normalize so cosine is well-defined.
        vec = [2.0 * v - 1.0 for v in values[: self._dimensions]]
        _l2_normalize_inplace(vec)
        return vec


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip — for stable hashing."""
    return " ".join(text.lower().split())


def _l2_normalize_inplace(vec: list[float]) -> None:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm < 1e-12:
        # Degenerate: leave as zeros (already zero-ish).
        for i in range(len(vec)):
            vec[i] = 0.0
        return
    for i in range(len(vec)):
        vec[i] = vec[i] / norm
