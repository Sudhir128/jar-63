"""Session manager: creates and tracks sessions (minimal implementation)."""

from __future__ import annotations

from typing import Any

from app.core.exceptions import JARError
from app.runtime.models import Session, SessionStatus

__all__ = ["SessionManager"]


class SessionManager:
    """In-memory session tracker.

    Only trivial bookkeeping is implemented; persistence belongs to a later
    phase.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def create_session(
        self,
        *,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        session = Session(user_id=user_id, metadata=metadata or {})
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Session:
        session = self._sessions.get(session_id)
        if session is None:
            raise JARError(f"Session not found: {session_id}")
        return session

    def list(self) -> list[Session]:
        return list(self._sessions.values())

    def close(self, session_id: str) -> Session:
        session = self.get(session_id)
        session = session.model_copy(update={"status": SessionStatus.CLOSED})
        self._sessions[session_id] = session
        return session

    def __len__(self) -> int:
        return len(self._sessions)
