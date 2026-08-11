"""Verify stage.

Evaluates objective evidence about the execution result. The verify stage is
the Checker in the Maker/Checker separation: it independently decides whether
the objective was achieved, regardless of whether execution "succeeded".

Phase 1 uses deterministic verifiers. The interface allows a future Checker
Agent to implement verification (e.g. running tests, checking build success,
validating required fields) without changing the loop engine.
"""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger
from app.runtime.loop.loop_context import LoopContext
from app.runtime.loop.loop_state import LoopStatus, StageStatus
from app.runtime.loop.stages.base import LoopStage
from app.runtime.loop.verification import (
    VerificationResult,
    Verifier,
)

logger = get_logger("loop.verify")

__all__ = ["VerifyStage", "DefaultVerifyStage"]


class VerifyStage(LoopStage):
    """Abstract verify stage contract."""

    name = "verify"

    async def run(self, context: LoopContext) -> LoopContext:
        result = await self.verify(context)
        context.update_state(
            current_stage=StageStatus.VERIFY,
            status=LoopStatus.VERIFYING,
            last_verification=result,
            verification_results=[*context.state.verification_results, result],
        )
        return context

    async def verify(self, context: LoopContext) -> VerificationResult:
        """Evaluate the last execution result and return evidence."""


class DefaultVerifyStage(VerifyStage):
    """Default verify stage.

    Uses an injected :class:`Verifier` (the checker). The expected value is
    read from the loop's success criteria / stage config. If verification is
    required by policy but no verifier is configured, the result is
    ``UNABLE_TO_VERIFY`` rather than an implicit success.
    """

    def __init__(self, verifier: Verifier | None = None) -> None:
        self._verifier = verifier

    async def verify(self, context: LoopContext) -> VerificationResult:
        last_exec = context.state.last_execution
        if last_exec is None:
            return VerificationResult.unable("No execution result to verify.")

        if last_exec.status != "success":
            return VerificationResult.failed_with(
                [],
                summary=f"Execution did not succeed (status={last_exec.status}); cannot verify objective.",
            )

        if self._verifier is None:
            if context.policy.require_verification:
                return VerificationResult.unable(
                    "Verification required but no verifier configured."
                )
            # When verification is optional and absent, we cannot claim
            # objective success — treat as unable rather than implicit pass.
            return VerificationResult.unable("No verifier configured.")

        expected = _resolve_expected(context)
        result = await self._verifier.verify(last_exec.output, expected=expected)
        return result


def _resolve_expected(context: LoopContext) -> Any:
    """Determine the expected value for verification.

    Priority: explicit stage config ``expected_output`` > first success
    criterion that encodes an expected value.
    """
    expected = context.stage_config.get("expected_output")
    if expected is not None:
        return expected
    for criterion in context.state.success_criteria:
        if isinstance(criterion, str) and criterion.startswith("expected:"):
            return criterion[len("expected:") :]
    return None
