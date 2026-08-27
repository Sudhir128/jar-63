"""Sanitization helpers for safe error reporting and event payloads.

Never blindly publish ``str(exception)`` — a tool exception may embed API
keys, passwords, or tokens. :func:`sanitize_error` redacts common
credential-patterns before a failure event payload is built. It is a safe
baseline, not a substitute for the broader secret-handling rules enforced
elsewhere (SecretStr masking, no prompt logging).
"""

from __future__ import annotations

import re

__all__ = ["sanitize_error", "REDACTED"]

REDACTED = "[redacted]"

# Matches common inline credential patterns:  api_key=..., "secret": "...",
# Authorization: Bearer <tok>, password=...  The <value> is captured and
# replaced without exposing it.
_KEYED_CREDENTIAL = re.compile(
    r"(api[_-]?key|secret|password|passwd|token|auth[_-]?token)\s*[:=]\s*"
    r"[\"']?[^\"',;\s}]+[\"']?",
    re.IGNORECASE,
)
_BEARER = re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]+", re.IGNORECASE)


_HEX_OR_B64_LIKE = re.compile(r"\b([0-9a-fA-F]{32,}|[A-Za-z0-9+\/_=]{24,})\b")


def sanitize_error(message: str, *, redact_tokens: bool = True) -> str:
    """Return a safe version of an error message with credentials redacted.

    Redacts key=value credential pairs (api_key/password/token/secret/...)
    and ``Bearer <token>``. With ``redact_tokens`` (default) it also redacts
    long hex/base64-like tokens, the most common accidental leaks.

    * ``redact_tokens=False`` disables the long-token heuristic for messages
      that may legitimately contain long non-secret identifiers.
    """
    if not message:
        return message
    safe = _KEYED_CREDENTIAL.sub(REDACTED, message)
    safe = _BEARER.sub(lambda m: f"{m.group(1)}{REDACTED}", safe)
    if redact_tokens:
        safe = _HEX_OR_B64_LIKE.sub(REDACTED, safe)
    return safe
