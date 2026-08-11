"""Verification result models.

The verifier is a first-class loop component. A loop may never declare
``SUCCESS`` without :class:`VerificationEvidence`.

Verification statuses:

* ``PASSED``           — every check passed.
* ``FAILED``           — at least one check failed.
* ``PARTIAL``          — some checks passed and some were unable to be verified.
* ``UNABLE_TO_VERIFY`` — no check could produce a definitive result.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.core.identifiers import generate_id, utc_now

__all__ = [
    "VerificationStatus",
    "VerificationEvidence",
    "VerificationResult",
]


class VerificationStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    PARTIAL = "partial"
    UNABLE_TO_VERIFY = "unable_to_verify"


class VerificationEvidence(BaseModel):
    """A single objective check produced by a verifier."""

    model_config = ConfigDict(frozen=True)

    check: str
    expected: Any = None
    actual: Any = None
    passed: bool = False
    detail: str | None = None


class VerificationResult(BaseModel):
    """The aggregated outcome of verifying an execution result."""

    model_config = ConfigDict(frozen=True)

    verification_id: str = Field(default_factory=lambda: generate_id("verify"))
    status: VerificationStatus
    evidence: list[VerificationEvidence] = Field(default_factory=list)
    summary: str = ""
    checked_at: datetime = Field(default_factory=utc_now)

    @property
    def passed(self) -> bool:
        """Convenience: whether verification succeeded."""
        return self.status is VerificationStatus.PASSED

    @property
    def has_evidence(self) -> bool:
        """Whether any objective evidence was produced."""
        return len(self.evidence) > 0

    @classmethod
    def passed_with(
        cls, evidence: list[VerificationEvidence], summary: str = ""
    ) -> VerificationResult:
        return cls(status=VerificationStatus.PASSED, evidence=evidence, summary=summary)

    @classmethod
    def failed_with(
        cls, evidence: list[VerificationEvidence], summary: str = ""
    ) -> VerificationResult:
        return cls(status=VerificationStatus.FAILED, evidence=evidence, summary=summary)

    @classmethod
    def unable(cls, summary: str = "") -> VerificationResult:
        return cls(status=VerificationStatus.UNABLE_TO_VERIFY, summary=summary)
