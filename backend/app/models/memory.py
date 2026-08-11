"""SQLAlchemy ORM models for the memory subsystem (Phase 6).

Tables:
    * ``memories``             — durable structured memory records.
    * ``conversation_messages`` — conversation history.
    * ``memory_relations``     — relationships between memories.

Indexes cover the common retrieval paths: user_id, session_id, task_id,
memory_type, created_at, updated_at.

These models are the persistence representation; the domain layer uses the
Pydantic :class:`~app.memory.models.MemoryRecord`. The two are translated at
the store boundary.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    Text as SQLText,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.identifiers import utc_now
from app.database import Base

__all__ = ["MemoryModel", "ConversationMessageModel", "MemoryRelationModel"]


class MemoryModel(Base):
    """Durable structured memory record (PostgreSQL source of truth)."""

    __tablename__ = "memories"

    memory_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    memory_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str] = mapped_column(Text, default="", server_default="")
    user_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    task_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    agent_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    source: Mapped[str] = mapped_column(String(32), default="system", server_default="system")
    importance: Mapped[float] = mapped_column(Float, default=0.5, server_default="0.5")
    confidence: Mapped[float] = mapped_column(Float, default=0.5, server_default="0.5")
    retention_policy: Mapped[str] = mapped_column(
        String(32), default="ttl", server_default="ttl"
    )
    # Embedding stored as a JSON array (works without pgvector).
    embedding: Mapped[list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, index=True
    )
    last_accessed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    access_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, default=dict, server_default="{}")

    relations_from: Mapped[list[MemoryRelationModel]] = relationship(
        "MemoryRelationModel",
        foreign_keys="MemoryRelationModel.from_memory_id",
        back_populates="from_memory",
    )
    relations_to: Mapped[list[MemoryRelationModel]] = relationship(
        "MemoryRelationModel",
        foreign_keys="MemoryRelationModel.to_memory_id",
        back_populates="to_memory",
    )

    __table_args__ = (
        Index("ix_memories_user_type", "user_id", "memory_type"),
        Index("ix_memories_session_type", "session_id", "memory_type"),
        Index("ix_memories_task_type", "task_id", "memory_type"),
    )


class ConversationMessageModel(Base):
    """Conversation history persistence."""

    __tablename__ = "conversation_messages"

    message_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    user_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    task_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    agent_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    tool_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, index=True
    )
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, default=dict, server_default="{}")

    __table_args__ = (
        Index("ix_conv_session_created", "session_id", "created_at"),
        Index("ix_conv_user_created", "user_id", "created_at"),
    )


class MemoryRelationModel(Base):
    """Relationships between memories (e.g. derives_from, supersedes)."""

    __tablename__ = "memory_relations"

    relation_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    from_memory_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("memories.memory_id", ondelete="CASCADE"), nullable=False, index=True
    )
    to_memory_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("memories.memory_id", ondelete="CASCADE"), nullable=False, index=True
    )
    relation_type: Mapped[str] = mapped_column(String(32), nullable=False)
    weight: Mapped[float] = mapped_column(Float, default=1.0, server_default="1.0")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    from_memory: Mapped[MemoryModel] = relationship(
        "MemoryModel", foreign_keys=[from_memory_id], back_populates="relations_from"
    )
    to_memory: Mapped[MemoryModel] = relationship(
        "MemoryModel", foreign_keys=[to_memory_id], back_populates="relations_to"
    )
