"""Tests for the LLM verifier extension point (not the default)."""

from __future__ import annotations

from app.llm.errors import LLMError
from app.llm.verifier import LLMVerifier
from app.runtime.loop.verification.verification_result import VerificationStatus
from tests.llm_helpers import StubLLMClient


async def test_llm_verifier_pass() -> None:
    client = StubLLMClient("ollama", content="pass\nthe output is correct")
    verifier = LLMVerifier(client, model="qwen2.5-coder:7b")
    result = await verifier.verify(output=42, expected=42)
    assert result.status is VerificationStatus.PASSED
    assert result.has_evidence
    assert result.evidence[0].passed is True


async def test_llm_verifier_fail() -> None:
    client = StubLLMClient("ollama", content="fail\nthe output is wrong")
    verifier = LLMVerifier(client, model="qwen2.5-coder:7b")
    result = await verifier.verify(output=41, expected=42)
    assert result.status is VerificationStatus.FAILED
    assert result.evidence[0].passed is False


async def test_llm_verifier_unable_on_error() -> None:
    client = StubLLMClient("ollama", raise_error=LLMError("boom"))
    verifier = LLMVerifier(client, model="qwen2.5-coder:7b")
    result = await verifier.verify(output=42, expected=42)
    assert result.status is VerificationStatus.UNABLE_TO_VERIFY


async def test_llm_verifier_produces_evidence() -> None:
    """The LLM verifier must still produce evidence (no SUCCESS without evidence)."""
    client = StubLLMClient("ollama", content="pass")
    verifier = LLMVerifier(client, model="qwen2.5-coder:7b")
    result = await verifier.verify(output="ok", expected="ok")
    assert len(result.evidence) == 1
    assert result.evidence[0].check == "llm_judge"
