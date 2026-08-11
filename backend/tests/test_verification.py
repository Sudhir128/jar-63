"""Tests for the verification subsystem."""

from __future__ import annotations

import pytest

from app.runtime.loop.verification import (
    CallableVerifier,
    CompositeVerifier,
    ExactMatchVerifier,
    VerificationStatus,
    always_fail_verifier,
    always_pass_verifier,
)


async def test_exact_match_verifier_passes() -> None:
    v = ExactMatchVerifier()
    result = await v.verify("hello", expected="hello")
    assert result.status is VerificationStatus.PASSED
    assert result.passed is True
    assert result.has_evidence is True
    assert len(result.evidence) == 1
    ev = result.evidence[0]
    assert ev.expected == "hello"
    assert ev.actual == "hello"
    assert ev.passed is True


async def test_exact_match_verifier_fails() -> None:
    v = ExactMatchVerifier()
    result = await v.verify(99, expected=100)
    assert result.status is VerificationStatus.FAILED
    assert result.passed is False
    assert result.evidence[0].actual == 99


async def test_always_pass_and_fail_verifiers() -> None:
    passed = await always_pass_verifier.verify("anything")
    failed = await always_fail_verifier.verify("anything")
    assert passed.status is VerificationStatus.PASSED
    assert failed.status is VerificationStatus.FAILED


async def test_callable_verifier_sync_predicate() -> None:
    v = CallableVerifier(lambda output: output > 10, check_name="gt_10")
    assert (await v.verify(42, expected=None)).status is VerificationStatus.PASSED
    assert (await v.verify(1, expected=None)).status is VerificationStatus.FAILED


async def test_callable_verifier_async_predicate() -> None:
    async def is_even(output: int) -> bool:
        return output % 2 == 0

    v = CallableVerifier(is_even, check_name="is_even")
    assert (await v.verify(4, expected=None)).status is VerificationStatus.PASSED
    assert (await v.verify(3, expected=None)).status is VerificationStatus.FAILED


async def test_callable_verifier_handles_exceptions() -> None:
    def boom(output: object) -> bool:
        raise ValueError("boom")

    v = CallableVerifier(boom)
    result = await v.verify("x", expected=None)
    assert result.status is VerificationStatus.UNABLE_TO_VERIFY
    assert "boom" in result.summary


async def test_composite_verifier_all_pass() -> None:
    v = CompositeVerifier([ExactMatchVerifier(), CallableVerifier(lambda o: o == "hello")])
    result = await v.verify("hello", expected="hello")
    assert result.status is VerificationStatus.PASSED
    assert len(result.evidence) == 2


async def test_composite_verifier_any_fail() -> None:
    v = CompositeVerifier([ExactMatchVerifier(), CallableVerifier(lambda o: o == "wrong")])
    result = await v.verify("hello", expected="hello")
    assert result.status is VerificationStatus.FAILED


def test_composite_verifier_requires_verifiers() -> None:
    with pytest.raises(ValueError):
        CompositeVerifier([])
