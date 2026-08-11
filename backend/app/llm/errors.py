"""Structured LLM error hierarchy.

All LLM errors extend :class:`~app.core.exceptions.JARError`. They are safe
for API responses — they never carry provider secrets, authorization headers,
or raw private user data. Provider-specific failure details are normalized
into these typed errors at the provider boundary.
"""

from __future__ import annotations

from app.core.exceptions import JARError

__all__ = [
    "LLMError",
    "ProviderUnavailableError",
    "ModelUnavailableError",
    "ModelNotFoundError",
    "LLMTimeoutError",
    "LLMAuthenticationError",
    "LLMRateLimitError",
    "InvalidLLMResponseError",
    "InvalidStructuredOutputError",
    "ModelRoutingError",
    "LLMPolicyError",
]


class LLMError(JARError):
    """Base class for all LLM-related errors."""


class ProviderUnavailableError(LLMError):
    """Raised when a provider cannot be reached or is not ready."""


class ModelUnavailableError(LLMError):
    """Raised when a requested model cannot be used (missing/disabled)."""


class ModelNotFoundError(LLMError):
    """Raised when a model is not found at a provider (e.g. not installed)."""


class LLMTimeoutError(LLMError):
    """Raised when an LLM request exceeds its timeout."""


class LLMAuthenticationError(LLMError):
    """Raised when a provider rejects authentication.

    Never includes the credential itself.
    """


class LLMRateLimitError(LLMError):
    """Raised when a provider rate-limits requests."""


class InvalidLLMResponseError(LLMError):
    """Raised when a provider response is malformed or unusable."""


class InvalidStructuredOutputError(LLMError):
    """Raised when structured output cannot be produced or fails validation."""


class ModelRoutingError(LLMError):
    """Raised when no model can be selected for a request."""


class LLMPolicyError(LLMError):
    """Raised when a request violates routing/privacy policy."""
