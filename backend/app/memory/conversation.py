"""Conversation store and summarizer (Phase 6).

The :class:`ConversationStore` persists conversation turns (to PostgreSQL and
optionally working memory). The :class:`ConversationSummarizer` produces a
deterministic summary of recent turns — it does **not** call an LLM. A future
phase can swap in an LLM-backed summarizer without changing callers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import select

from app.core.logging import get_logger
from app.database import AsyncSessionLocal
from app.memory.models import ConversationMessage
from app.models.memory import ConversationMessageModel

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


logger = get_logger("memory.conversation")

__all__ = ["ConversationStore", "ConversationSummarizer", "summarize_turns"]


class ConversationStore:
    """Persistence for conversation history (PostgreSQL)."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
    ) -> None:
        self._session_factory = session_factory or AsyncSessionLocal

    async def add(self, message: ConversationMessage) -> ConversationMessage:
        async with self._session_factory() as session:
            row = ConversationMessageModel(
                message_id=message.message_id,
                role=message.role,
                content=message.content,
                user_id=message.user_id,
                session_id=message.session_id,
                task_id=message.task_id,
                agent_id=message.agent_id,
                tool_name=message.tool_name,
                created_at=message.created_at,
                metadata_=dict(message.metadata),
            )
            session.add(row)
            await session.commit()
            return message

    async def list_recent(
        self,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        limit: int = 20,
    ) -> list[ConversationMessage]:
        stmt = select(ConversationMessageModel)
        if session_id is not None:
            stmt = stmt.where(ConversationMessageModel.session_id == session_id)
        if user_id is not None:
            stmt = stmt.where(ConversationMessageModel.user_id == user_id)
        stmt = stmt.order_by(ConversationMessageModel.created_at.desc()).limit(limit)
        async with self._session_factory() as session:
            rows = (await session.execute(stmt)).scalars().all()
        messages = [_row_to_message(r) for r in rows]
        messages.reverse()  # chronological order
        return messages

    async def count(
        self,
        *,
        session_id: str | None = None,
    ) -> int:
        from sqlalchemy import func

        stmt = select(func.count()).select_from(ConversationMessageModel)
        if session_id is not None:
            stmt = stmt.where(ConversationMessageModel.session_id == session_id)
        async with self._session_factory() as session:
            result = await session.execute(stmt)
            return int(result.scalar() or 0)


class ConversationSummarizer:
    """Deterministic conversation summarizer (no LLM).

    Extracts user turns and produces a compact bullet summary. This is a
    placeholder summarizer: it captures the structure of a conversation
    without semantic compression. A future phase can replace it with an
    LLM-backed summarizer implementing the same interface.
    """

    def __init__(self, max_turns: int = 5, max_chars: int = 500) -> None:
        self._max_turns = max_turns
        self._max_chars = max_chars

    def summarize(self, messages: list[ConversationMessage]) -> str:
        if not messages:
            return ""
        user_turns = [m for m in messages if m.role == "user"]
        recent = user_turns[-self._max_turns :]
        if not recent:
            recent = messages[-self._max_turns :]
        lines: list[str] = []
        total = 0
        for m in recent:
            content = (m.content or "").strip().replace("\n", " ")
            if not content:
                continue
            line = f"- {content}"
            if total + len(line) > self._max_chars:
                break
            lines.append(line)
            total += len(line)
        if not lines:
            return ""
        header = f"Summary of {len(messages)} message(s):"
        return header + "\n" + "\n".join(lines)


def summarize_turns(messages: list[ConversationMessage]) -> str:
    """Convenience function: summarize a list of conversation turns."""
    return ConversationSummarizer().summarize(messages)


def _row_to_message(row: ConversationMessageModel) -> ConversationMessage:
    return ConversationMessage(
        message_id=row.message_id,
        role=row.role,
        content=row.content,
        user_id=row.user_id,
        session_id=row.session_id,
        task_id=row.task_id,
        agent_id=row.agent_id,
        tool_name=row.tool_name,
        created_at=row.created_at,
        metadata=dict(row.metadata_ or {}),
    )
