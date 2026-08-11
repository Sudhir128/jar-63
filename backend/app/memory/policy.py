"""Memory write policy: privacy guardrail and duplicate detection (Phase 6).

Before a memory is persisted, the write policy decides what to do:

1. **Privacy check** — reject content that appears to contain secrets/PII
   (configurable substring patterns) unless explicitly allowed.
2. **Duplicate detection** — if a near-duplicate already exists (same type +
   scope, high text similarity), decide whether to update the existing
   record or ignore the new one.

The decision is one of :class:`MemoryWriteDecision` (STORE / UPDATE / IGNORE
/ EXPIRE). The policy is intentionally simple and deterministic; a future
phase can add LLM-based deduplication or a learned privacy classifier.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass

from app.config import MemorySettings
from app.memory.models import MemoryRecord
from app.memory.types import MemoryWriteDecision

__all__ = ["MemoryWritePolicy", "DuplicateCheck", "WriteDecision", "SENSITIVE_HINT"]


@dataclass(frozen=True)
class WriteDecision:
    """The outcome of evaluating a candidate memory."""

    decision: MemoryWriteDecision
    existing_id: str | None = None
    reason: str = ""


@dataclass
class DuplicateCheck:
    """Result of a duplicate-detection lookup."""

    is_duplicate: bool
    existing_id: str | None = None
    similarity: float = 0.0


# Hint shown in logs/events when privacy blocks a write (never the content).
SENSITIVE_HINT = "sensitive_pattern_detected"


class MemoryWritePolicy:
    """Decides whether/how to persist a candidate :class:`MemoryRecord`."""

    def __init__(self, settings: MemorySettings | None = None) -> None:
        self._settings = settings or MemorySettings()
        self._threshold = self._settings.duplicate_similarity_threshold
        patterns = [p.lower() for p in self._settings.sensitive_patterns]
        self._sensitive = patterns
        self._allow_sensitive = self._settings.allow_sensitive_storage

    def evaluate(
        self,
        candidate: MemoryRecord,
        existing: list[MemoryRecord],
    ) -> WriteDecision:
        """Evaluate a candidate against privacy rules and existing memories."""
        # 1. Privacy guardrail.
        if not self._allow_sensitive:
            blocked = self._detect_sensitive(candidate.content)
            if blocked:
                return WriteDecision(
                    decision=MemoryWriteDecision.IGNORE,
                    reason=SENSITIVE_HINT,
                )

        # 2. Duplicate detection (same type + overlapping scope).
        dup = self._find_duplicate(candidate, existing)
        if dup.is_duplicate:
            # If the candidate has higher confidence, update; else ignore.
            existing_rec = next((m for m in existing if m.memory_id == dup.existing_id), None)
            if existing_rec is None:
                return WriteDecision(
                    decision=MemoryWriteDecision.UPDATE,
                    existing_id=dup.existing_id,
                    reason="duplicate_found",
                )
            if candidate.confidence > existing_rec.confidence + 0.05:
                return WriteDecision(
                    decision=MemoryWriteDecision.UPDATE,
                    existing_id=dup.existing_id,
                    reason="higher_confidence_duplicate",
                )
            return WriteDecision(
                decision=MemoryWriteDecision.IGNORE,
                existing_id=dup.existing_id,
                reason="duplicate_exists",
            )

        # 3. Expiry for short-lived types.
        if candidate.memory_type.is_long_lived is False and candidate.expires_at is None:
            return WriteDecision(
                decision=MemoryWriteDecision.EXPIRE,
                reason="apply_default_retention",
            )

        return WriteDecision(decision=MemoryWriteDecision.STORE, reason="new_memory")

    def _find_duplicate(
        self, candidate: MemoryRecord, existing: list[MemoryRecord]
    ) -> DuplicateCheck:
        """Find the most similar existing record within the same scope/type."""
        best_id: str | None = None
        best_sim = 0.0
        for rec in existing:
            if rec.memory_type != candidate.memory_type:
                continue
            if not _same_scope(rec, candidate):
                continue
            sim = _text_similarity(candidate.content, rec.content)
            if sim > best_sim:
                best_sim = sim
                best_id = rec.memory_id
        return DuplicateCheck(
            is_duplicate=best_sim >= self._threshold,
            existing_id=best_id,
            similarity=best_sim,
        )

    def _detect_sensitive(self, content: str) -> bool:
        """Return True if ``content`` matches a configured sensitive pattern."""
        if not content or not self._sensitive:
            return False
        lower = content.lower()
        return any(p in lower for p in self._sensitive)


def _same_scope(a: MemoryRecord, b: MemoryRecord) -> bool:
    """Whether two records share a meaningful scope for dedup."""
    return not (a.user_id is not None and b.user_id is not None and a.user_id != b.user_id)


def _text_similarity(a: str, b: str) -> float:
    """Deterministic text similarity in [0.0, 1.0] using SequenceMatcher."""
    if not a or not b:
        return 0.0
    norm_a = " ".join(a.lower().split())
    norm_b = " ".join(b.lower().split())
    return difflib.SequenceMatcher(None, norm_a, norm_b).ratio()
