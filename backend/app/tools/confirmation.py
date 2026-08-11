"""Confirmation state for tools requiring user approval.

When the :class:`~app.tools.policy.ToolPolicy` returns
``REQUIRE_CONFIRMATION``, the runtime pauses execution and creates a
:class:`ConfirmationRequest`. The request is stored in the loop state and an
event is published. An API endpoint (or the future Voice layer) can then
approve or reject the action, resuming the loop.

This is the foundation — the confirmation store is in-memory for Phase 3.
Later phases may persist confirmations and add richer authorization.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.core.identifiers import generate_id, utc_now
from app.tools.interface import RiskLevel

__all__ = [
    "ConfirmationStatus",
    "ConfirmationRequest",
    "ConfirmationStore",
    "DEFAULT_CONFIRMATION_TTL_SECONDS",
]

DEFAULT_CONFIRMATION_TTL_SECONDS = 600


class ConfirmationStatus(StrEnum):
    """Lifecycle status of a confirmation request."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ConfirmationRequest(BaseModel):
    """A request for user confirmation of a tool call.

    Stored in the loop state when a tool requires confirmation. The loop
    pauses until the request is approved, rejected, or expired.
    """

    model_config = ConfigDict(frozen=True)

    confirmation_id: str = Field(default_factory=lambda: generate_id("confirm"))
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    risk_level: RiskLevel = RiskLevel.LOW
    task_id: str | None = None
    session_id: str | None = None
    loop_id: str | None = None
    iteration: int | None = None
    tool_call_id: str | None = None
    reason: str = ""
    status: ConfirmationStatus = ConfirmationStatus.PENDING
    created_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime | None = None
    decided_at: datetime | None = None
    decided_by: str | None = None

    @property
    def is_pending(self) -> bool:
        return self.status is ConfirmationStatus.PENDING

    @property
    def is_approved(self) -> bool:
        return self.status is ConfirmationStatus.APPROVED

    @property
    def is_rejected(self) -> bool:
        return self.status is ConfirmationStatus.REJECTED

    @property
    def is_expired(self) -> bool:
        return self.status is ConfirmationStatus.EXPIRED

    @property
    def has_expired(self) -> bool:
        """Whether the confirmation has passed its TTL (lazy check)."""
        if self.status is not ConfirmationStatus.PENDING:
            return False
        if self.expires_at is None:
            return False
        return utc_now() >= self.expires_at


class ConfirmationStore:
    """In-memory store of pending/decided confirmation requests.

    The store is the single mechanism through which confirmations are created,
    approved, rejected, and expired. Later phases may back this with Redis or
    the DB. Expiration is checked lazily on access — no background polling.
    """

    def __init__(self, *, default_ttl_seconds: int = DEFAULT_CONFIRMATION_TTL_SECONDS) -> None:
        self._requests: dict[str, ConfirmationRequest] = {}
        self._default_ttl = default_ttl_seconds

    def create(
        self, request: ConfirmationRequest, *, ttl_seconds: int | None = None
    ) -> ConfirmationRequest:
        if request.expires_at is None:
            ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
            request = request.model_copy(
                update={"expires_at": request.created_at + timedelta(seconds=ttl)}
            )
        self._requests[request.confirmation_id] = request
        return request

    def get(self, confirmation_id: str) -> ConfirmationRequest | None:
        req = self._requests.get(confirmation_id)
        if req is not None and req.has_expired:
            req = self._expire(req)
        return req

    def approve(
        self, confirmation_id: str, *, decided_by: str | None = None
    ) -> ConfirmationRequest | None:
        req = self._requests.get(confirmation_id)
        if req is None:
            return None
        if req.has_expired:
            self._expire(req)
            return None
        if not req.is_pending:
            return None
        updated = req.model_copy(
            update={
                "status": ConfirmationStatus.APPROVED,
                "decided_at": utc_now(),
                "decided_by": decided_by,
            }
        )
        self._requests[confirmation_id] = updated
        return updated

    def reject(
        self, confirmation_id: str, *, decided_by: str | None = None
    ) -> ConfirmationRequest | None:
        req = self._requests.get(confirmation_id)
        if req is None:
            return None
        if req.has_expired:
            self._expire(req)
            return None
        if not req.is_pending:
            return None
        updated = req.model_copy(
            update={
                "status": ConfirmationStatus.REJECTED,
                "decided_at": utc_now(),
                "decided_by": decided_by,
            }
        )
        self._requests[confirmation_id] = updated
        return updated

    def expire(self, confirmation_id: str) -> ConfirmationRequest | None:
        """Explicitly expire a pending confirmation."""
        req = self._requests.get(confirmation_id)
        if req is None or not req.is_pending:
            return None
        return self._expire(req)

    def list_pending(self) -> list[ConfirmationRequest]:
        # Lazily expire any pending requests whose TTL has elapsed.
        for cid, req in list(self._requests.items()):
            if req.has_expired:
                self._expire(req)
        return [r for r in self._requests.values() if r.is_pending]

    def __len__(self) -> int:
        return len(self._requests)

    def _expire(self, req: ConfirmationRequest) -> ConfirmationRequest:
        if not req.is_pending:
            return req
        updated = req.model_copy(
            update={"status": ConfirmationStatus.EXPIRED, "decided_at": utc_now()}
        )
        self._requests[req.confirmation_id] = updated
        return updated
