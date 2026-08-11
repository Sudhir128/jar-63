"""Tests for the LLM error hierarchy."""

from __future__ import annotations

from app.core.exceptions import JARError
from app.llm.errors import (
    InvalidLLMResponseError,
    InvalidStructuredOutputError,
    LLMAuthenticationError,
    LLMError,
    LLMPolicyError,
    LLMRateLimitError,
    LLMTimeoutError,
    ModelNotFoundError,
    ModelRoutingError,
    ModelUnavailableError,
    ProviderUnavailableError,
)


def test_all_errors_extend_llm_error() -> None:
    for exc in [
        ProviderUnavailableError,
        ModelUnavailableError,
        ModelNotFoundError,
        LLMTimeoutError,
        LLMAuthenticationError,
        LLMRateLimitError,
        InvalidLLMResponseError,
        InvalidStructuredOutputError,
        ModelRoutingError,
        LLMPolicyError,
    ]:
        assert issubclass(exc, LLMError)
        assert issubclass(exc, JARError)


def test_llm_error_is_jar_error() -> None:
    assert issubclass(LLMError, JARError)


def test_errors_carry_safe_messages() -> None:
    err = LLMAuthenticationError("auth failed (status 401)")
    assert "auth failed" in str(err)
    # No secret fields by construction.
    assert not hasattr(err, "api_key")
