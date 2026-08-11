"""Loop engine error hierarchy.

These errors describe controlled failures within the Universal Loop Engine.
They never indicate uncontrolled autonomy — every failure maps to one of the
observable :class:`~app.runtime.loop.loop_result.LoopFinalStatus` values.
"""

from __future__ import annotations

from app.core.exceptions import JARError

__all__ = [
    "LoopError",
    "LoopPolicyError",
    "LoopStageError",
    "LoopTimeoutError",
    "LoopCancelledError",
    "LoopMaxIterationsError",
    "VerificationError",
]


class LoopError(JARError):
    """Base class for all loop-engine errors."""


class LoopPolicyError(LoopError):
    """Raised when a loop is started with an invalid policy."""


class LoopStageError(LoopError):
    """Raised when a stage fails to produce a result."""


class LoopTimeoutError(LoopError):
    """Raised when execution exceeds the configured time budget."""


class LoopCancelledError(LoopError):
    """Raised when a loop is cancelled while running."""


class LoopMaxIterationsError(LoopError):
    """Raised internally when the hard iteration limit is reached."""


class VerificationError(LoopError):
    """Raised when verification is required but cannot be performed."""
