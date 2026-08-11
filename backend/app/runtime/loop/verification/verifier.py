"""Verifier interface and deterministic default verifiers.

The verifier evaluates objective evidence about an execution result. It is
deliberately decoupled from the execute stage so that a future Checker Agent
can implement :class:`Verifier` without changing the loop engine.

Maker / Checker separation: the entity that *produces* work (an execute agent)
must not be assumed correct. The :class:`Verifier` (the checker) independently
evaluates the result.
"""

from __future__ import annotations

import abc
from collections.abc import Awaitable, Callable
from typing import Any

from app.core.logging import get_logger
from app.runtime.loop.verification.verification_result import (
    VerificationEvidence,
    VerificationResult,
    VerificationStatus,
)

logger = get_logger("loop.verifier")

__all__ = [
    "Verifier",
    "ExactMatchVerifier",
    "CallableVerifier",
    "CompositeVerifier",
    "always_pass_verifier",
    "always_fail_verifier",
]

# A synchronous or asynchronous predicate over an execution output.
CheckFn = Callable[[Any], bool] | Callable[[Any], Awaitable[bool]]


class Verifier(abc.ABC):
    """Abstract verifier contract.

    A verifier inspects the execution ``output`` (and optionally the loop
    context) and returns a :class:`VerificationResult` backed by evidence.
    """

    @abc.abstractmethod
    async def verify(self, output: Any, *, expected: Any = None) -> VerificationResult:
        """Evaluate ``output`` against ``expected`` and return evidence."""


class ExactMatchVerifier(Verifier):
    """Verify that the output equals an expected value exactly."""

    def __init__(self, check_name: str = "exact_match") -> None:
        self._check_name = check_name

    async def verify(self, output: Any, *, expected: Any = None) -> VerificationResult:
        passed = output == expected
        evidence = VerificationEvidence(
            check=self._check_name,
            expected=expected,
            actual=output,
            passed=passed,
            detail="output matches expected value"
            if passed
            else "output differs from expected value",
        )
        status = VerificationStatus.PASSED if passed else VerificationStatus.FAILED
        return VerificationResult(
            status=status,
            evidence=[evidence],
            summary=f"exact match: {output!r} == {expected!r}"
            if passed
            else f"exact match failed: {output!r} != {expected!r}",
        )


class CallableVerifier(Verifier):
    """Verify using a user-supplied predicate function.

    The predicate may be sync or async and must return a boolean. This lets
    later phases plug in custom checks (test runs, build success, exit codes)
    without subclassing :class:`Verifier`.
    """

    def __init__(self, check_fn: CheckFn, check_name: str = "predicate") -> None:
        self._check_fn = check_fn
        self._check_name = check_name

    async def verify(self, output: Any, *, expected: Any = None) -> VerificationResult:
        try:
            result = self._check_fn(output)
            if hasattr(result, "__await__"):
                result = await result  # type: ignore[func-returns-value]
            passed = bool(result)
        except Exception as exc:  # noqa: BLE001 - a failing check is evidence, not a crash
            logger.bind(event="verifier.error", error=type(exc).__name__).warning(
                "Callable verifier raised: {}", str(exc)
            )
            return VerificationResult(
                status=VerificationStatus.UNABLE_TO_VERIFY,
                evidence=[
                    VerificationEvidence(
                        check=self._check_name,
                        expected=expected,
                        actual=output,
                        passed=False,
                        detail=f"verifier raised: {exc}",
                    )
                ],
                summary=f"verifier raised: {exc}",
            )
        evidence = VerificationEvidence(
            check=self._check_name, expected=expected, actual=output, passed=passed
        )
        status = VerificationStatus.PASSED if passed else VerificationStatus.FAILED
        return VerificationResult(
            status=status,
            evidence=[evidence],
            summary=f"{self._check_name}: {passed}",
        )


class CompositeVerifier(Verifier):
    """Run multiple verifiers and aggregate their evidence.

    Aggregation rules:

    * all checks passed -> ``PASSED``
    * any check failed   -> ``FAILED``
    * otherwise (no failures, some unable) -> ``PARTIAL``
    * no definitive evidence -> ``UNABLE_TO_VERIFY``
    """

    def __init__(self, verifiers: list[Verifier]) -> None:
        if not verifiers:
            raise ValueError("CompositeVerifier requires at least one verifier")
        self._verifiers = verifiers

    async def verify(self, output: Any, *, expected: Any = None) -> VerificationResult:
        all_evidence: list[VerificationEvidence] = []
        any_failed = False
        for v in self._verifiers:
            res = await v.verify(output, expected=expected)
            all_evidence.extend(res.evidence)
            if res.status is VerificationStatus.FAILED:
                any_failed = True
        if any_failed:
            status = VerificationStatus.FAILED
        elif all_evidence and not any_failed:
            status = (
                VerificationStatus.PASSED
                if all(e.passed for e in all_evidence)
                else VerificationStatus.PARTIAL
            )
        else:
            status = VerificationStatus.UNABLE_TO_VERIFY
        return VerificationResult(
            status=status, evidence=all_evidence, summary=f"composite: {status.value}"
        )


class _AlwaysPassVerifier(Verifier):
    async def verify(self, output: Any, *, expected: Any = None) -> VerificationResult:
        return VerificationResult.passed_with(
            [
                VerificationEvidence(
                    check="always_pass", expected=expected, actual=output, passed=True
                )
            ],
            summary="always passes",
        )


class _AlwaysFailVerifier(Verifier):
    async def verify(self, output: Any, *, expected: Any = None) -> VerificationResult:
        return VerificationResult.failed_with(
            [
                VerificationEvidence(
                    check="always_fail", expected=expected, actual=output, passed=False
                )
            ],
            summary="always fails",
        )


always_pass_verifier = _AlwaysPassVerifier()
always_fail_verifier = _AlwaysFailVerifier()
