"""Small shared utilities (IDs, time)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

__all__ = ["utc_now", "generate_id"]


def utc_now() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(UTC)


def generate_id(prefix: str | None = None) -> str:
    """Return a new unique identifier, optionally prefixed."""
    value = uuid.uuid4().hex
    return f"{prefix}_{value}" if prefix else value
