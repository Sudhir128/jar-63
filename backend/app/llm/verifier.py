"""LLM-based verifier (extension point, NOT the default).

Phase 2 keeps objective verifiers (:class:`~app.runtime.loop.verification.Verifier`)
as the first choice. This module provides the extension point for an LLM-based
verifier so that future phases can verify objectives that cannot be checked
deterministically (e.g. "is this summary coherent?").

It is **never** wired as the default verifier. The Phase 1 principle holds:
no SUCCESS without evidence. When the LLM verifier is used, it must still
produce :class:`VerificationEvidence`.
"""

from __future__ import annotations

from typing import Any

from app.llm.client import LLMClient
from app.llm.errors import LLMError
from app.llm.models import LLMMessage, LLMRequest, MessageRole, PrivacyLevel
from app.runtime.loop.verification.verification_result import (
    VerificationEvidence,
    VerificationResult,
    VerificationStatus,
)
from app.runtime.loop.verification.verifier import Verifier

__all__ = ["LLMVerifier"]


class LLMVerifier(Verifier):
    """Verifies an output using an LLM judge.

    The LLM is asked to return ``pass`` or ``fail`` plus a reason. The result
    is wrapped in :class:`VerificationEvidence` so the loop engine's
    "no SUCCESS without evidence" rule is preserved.

    On any LLM failure, the result is ``UNABLE_TO_VERIFY`` (never an implicit
    pass).
    """

    def __init__(
        self,
        client: LLMClient,
        model: str,
        *,
        check_name: str = "llm_judge",
    ) -> None:
        self._client = client
        self._model = model
        self._check_name = check_name

    async def verify(self, output: Any, *, expected: Any = None) -> VerificationResult:
        prompt = (
            f"You are an objective verifier. Determine whether the output satisfies "
            f"the objective.\n\nObjective/expected: {expected}\n\nOutput: {output}\n\n"
            f"Respond with exactly 'pass' or 'fail' on the first line, then a one-line reason."
        )
        request = LLMRequest(
            model=self._model,
            messages=[
                LLMMessage(role=MessageRole.SYSTEM, content="You verify objective success."),
                LLMMessage(role=MessageRole.USER, content=prompt),
            ],
            temperature=0.0,
            privacy=PrivacyLevel.INTERNAL,
        )
        try:
            response = await self._client.generate(request)
        except LLMError:
            return VerificationResult.unable("LLM verifier request failed.")
        content = response.content.strip().lower()
        passed = content.startswith("pass")
        return VerificationResult(
            status=VerificationStatus.PASSED if passed else VerificationStatus.FAILED,
            evidence=[
                VerificationEvidence(
                    check=self._check_name,
                    expected=expected,
                    actual=output,
                    passed=passed,
                    detail=response.content[:200],
                )
            ],
            summary=f"llm_judge: {'pass' if passed else 'fail'}",
        )
