"""Memory taxonomy: types, sources, retention, and write-policy enums.

These enums define the vocabulary of the memory subsystem. They are used
by domain models (:mod:`app.memory.models`), the store layer, the write
policy, and the retriever.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "MemoryType",
    "MemorySource",
    "RetentionPolicy",
    "MemoryWriteDecision",
    "MEMORY_TYPE_DEFAULT_RETENTION",
]


class MemoryType(StrEnum):
    """Typed memory taxonomy.

    * ``WORKING``      — temporary, current-loop/task context (short-lived).
    * ``CONVERSATION`` — recent conversational turns (limited retention).
    * ``EPISODIC``     — events/experiences that happened.
    * ``SEMANTIC``     — stable factual knowledge (long-lived).
    * ``PREFERENCE``   — explicit user preferences (long-lived).
    * ``TASK``         — information associated with a task.
    * ``AGENT``        — agent/workflow-specific memory (isolated).
    """

    WORKING = "working"
    CONVERSATION = "conversation"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PREFERENCE = "preference"
    TASK = "task"
    AGENT = "agent"

    @property
    def is_durable(self) -> bool:
        """Whether this type is normally persisted to PostgreSQL."""
        return self is not MemoryType.WORKING

    @property
    def is_long_lived(self) -> bool:
        """Whether this type has no default expiry."""
        return self in (MemoryType.SEMANTIC, MemoryType.PREFERENCE)


class MemorySource(StrEnum):
    """How a memory was created (trust/retention signal)."""

    USER_EXPLICIT = "user_explicit"
    CONVERSATION = "conversation"
    AGENT = "agent"
    TOOL = "tool"
    SYSTEM = "system"
    IMPORTED = "imported"


class RetentionPolicy(StrEnum):
    """Retention strategy for a memory record."""

    PERMANENT = "permanent"
    TTL = "ttl"
    TASK_LIFETIME = "task_lifetime"
    SESSION_LIFETIME = "session_lifetime"


class MemoryWriteDecision(StrEnum):
    """Decision returned by :class:`MemoryWritePolicy`."""

    STORE = "store"
    UPDATE = "update"
    IGNORE = "ignore"
    EXPIRE = "expire"


# Default retention (hours) per memory type. None = no expiry (permanent).
MEMORY_TYPE_DEFAULT_RETENTION: dict[MemoryType, int | None] = {
    MemoryType.WORKING: 1,
    MemoryType.CONVERSATION: 48,
    MemoryType.EPISODIC: None,
    MemoryType.SEMANTIC: None,
    MemoryType.PREFERENCE: None,
    MemoryType.TASK: 168,
    MemoryType.AGENT: 168,
}
