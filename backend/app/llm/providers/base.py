"""Shared helpers for LLM provider implementations."""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger
from app.llm.errors import (
    InvalidLLMResponseError,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMTimeoutError,
    ProviderUnavailableError,
)
from app.llm.models import LLMRequest

logger = get_logger("llm.providers")

__all__ = ["BaseProviderMixin", "normalize_timeout"]


def normalize_timeout(request: LLMRequest, default: float) -> float:
    """Return the effective request timeout (request override > default)."""
    if request.timeout is not None:
        return float(request.timeout)
    return default


class BaseProviderMixin:
    """Common provider behavior: HTTP error normalization + safe logging.

    Provider implementations mix this in to map transport/HTTP errors onto the
    typed LLM error hierarchy without leaking secrets.
    """

    provider_name: str = "base"

    @staticmethod
    def _classify_http_status(status: int, *, provider: str) -> Exception:
        if status == 401 or status == 403:
            return LLMAuthenticationError(f"{provider}: authentication failed (status {status})")
        if status == 429:
            return LLMRateLimitError(f"{provider}: rate limited (status 429)")
        if status >= 500:
            return ProviderUnavailableError(f"{provider}: server error (status {status})")
        return InvalidLLMResponseError(f"{provider}: unexpected status {status}")

    @staticmethod
    def _classify_transport_error(exc: BaseException, *, provider: str) -> Exception:
        import httpx

        if isinstance(exc, httpx.TimeoutException):
            return LLMTimeoutError(f"{provider}: request timed out")
        if isinstance(exc, httpx.ConnectError):
            return ProviderUnavailableError(f"{provider}: connection failed")
        if isinstance(exc, httpx.HTTPError):
            return ProviderUnavailableError(f"{provider}: transport error: {type(exc).__name__}")
        return InvalidLLMResponseError(f"{provider}: {type(exc).__name__}: {exc}")

    def _log_request(self, request: LLMRequest, *, verbose: bool) -> None:
        """Log a request safely (no prompt contents unless verbose dev mode)."""
        bound = logger.bind(
            provider=self.provider_name,
            model=request.model,
            request_id=request.request_id,
            privacy=request.privacy.value,
        )
        if verbose:
            bound.debug("LLM request (verbose)", messages=request.messages)
        else:
            bound.debug("LLM request ({} messages)", len(request.messages))

    def _log_response(
        self,
        model: str,
        provider: str,
        *,
        latency_ms: int,
        success: bool,
        verbose: bool,
        content: str = "",
    ) -> None:
        bound = logger.bind(provider=provider, model=model, latency_ms=latency_ms, success=success)
        if verbose and content:
            bound.debug("LLM response (verbose)", content_preview=content[:200])
        else:
            bound.debug("LLM response success={} latency_ms={}", success, latency_ms)

    @staticmethod
    def _safe_metadata(**kwargs: Any) -> dict[str, Any]:
        """Build a metadata dict, never including secret values."""
        return dict(kwargs)
