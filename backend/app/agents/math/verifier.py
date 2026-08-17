"""Math verification: objective verification of arithmetic results.

The verifier independently re-computes the expected result from the expression
in the agent's output and checks it matches the claimed result. It does not
trust the LLM — it uses the same safe AST evaluator the calculator uses, but
as an independent check.

This is the Checker in the Maker/Checker separation: the MathAgent (maker)
produces the answer, and this verifier (checker) independently confirms it.
"""

from __future__ import annotations

from typing import Any

from app.runtime.loop.verification import VerificationResult
from app.runtime.loop.verification.verification_result import VerificationEvidence
from app.runtime.loop.verification.verifier import Verifier
from app.tools.impl import _CalcError, _safe_eval

__all__ = ["MathVerifier"]


class MathVerifier(Verifier):
    """Objectively verifies a MathAgent result by re-computing the expression.

    Extracts the expression and claimed result from the agent's output dict,
    independently evaluates the expression using the safe AST evaluator, and
    checks the results match. This is independent of the LLM — it is a pure
    mathematical check.
    """

    async def verify(self, actual: Any, expected: Any = None) -> VerificationResult:
        if not isinstance(actual, dict):
            return VerificationResult.failed_with(
                [],
                summary="Math result is not a dict; cannot verify.",
            )
        expression = actual.get("expression")
        claimed = actual.get("result")
        if not expression or claimed is None:
            return VerificationResult.failed_with(
                [],
                summary="Math result missing 'expression' or 'result' field.",
            )

        try:
            independent = _safe_eval(str(expression))
        except _CalcError as exc:
            return VerificationResult.failed_with(
                [
                    VerificationEvidence(
                        check="independent_recompute",
                        expected="computable expression",
                        actual=str(exc),
                        passed=False,
                        detail="Expression could not be independently evaluated.",
                    )
                ],
                summary=f"Verification failed: {exc}",
            )
        except Exception as exc:  # noqa: BLE001
            return VerificationResult.failed_with(
                [
                    VerificationEvidence(
                        check="independent_recompute",
                        expected="computable expression",
                        actual=str(exc),
                        passed=False,
                    )
                ],
                summary=f"Verification failed: {exc}",
            )

        # Compare numerically (handle int vs float equivalence).
        passed = _numeric_equal(independent, claimed)
        evidence = VerificationEvidence(
            check="independent_recompute",
            expected=independent,
            actual=claimed,
            passed=passed,
            detail=(
                f"Independently computed {independent} from '{expression}'; "
                f"agent claimed {claimed}."
            ),
        )
        if passed:
            return VerificationResult.passed_with(
                [evidence],
                summary=f"Result {claimed} verified independently (={independent}).",
            )
        return VerificationResult.failed_with(
            [evidence],
            summary=f"Result mismatch: expected {independent}, got {claimed}.",
        )


def _numeric_equal(a: Any, b: Any) -> bool:
    """Compare two numeric values, allowing int/float equivalence."""
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return a == b
