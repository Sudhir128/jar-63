"""Structured logging foundation.

Built on Loguru with support for:
* context fields (component, event, task_id, agent_id, session_id, error)
* JSON serialization for production
* secret/credential masking
* console (dev) and file sinks (opt-in)
"""

from __future__ import annotations

import logging
import re
import sys
from types import TracebackType
from typing import Any

from loguru import logger

from app.config import LoggingSettings

__all__ = ["configure_logging", "get_logger", "logger"]

# Patterns whose values must never appear in logs.
_SECRET_PATTERNS = [
    re.compile(r"(?i)(password|passwd|secret|token|api[_-]?key|authorization)\b"),
]
# Mask the bearer/token portion of common Authorization header formats.
_BEARER_RE = re.compile(r"(?i)(bearer\s+)([A-Za-z0-9._\-]+)")
# Mask password components embedded in URLs: scheme://user:pass@host
_URL_PASS_RE = re.compile(r"(://[^:/@\s]+:)([^@/\s]+)(@)")


def _mask_value(key: str, value: Any) -> Any:
    """Return a masked representation of a value when its key looks sensitive."""
    if value is None:
        return None
    if isinstance(value, str):
        if _BEARER_RE.search(value):
            value = _BEARER_RE.sub(r"\1<redacted>", value)
        if _URL_PASS_RE.search(value):
            value = _URL_PASS_RE.sub(r"\1<redacted>\3", value)
        if any(p.search(key) for p in _SECRET_PATTERNS):
            return "<redacted>"
    return value


def _mask_record(record: dict[str, Any]) -> dict[str, Any]:
    """Mask sensitive values inside the ``extra`` payload of a log record."""
    extra = record.get("extra")
    if isinstance(extra, dict):
        record["extra"] = {k: _mask_value(k, v) for k, v in extra.items()}
    message = record.get("message")
    if isinstance(message, str):
        record["message"] = _URL_PASS_RE.sub(r"\1<redacted>\3", message)
    return record


def configure_logging(settings: LoggingSettings | None = None) -> None:
    """Configure the global Loguru logger.

    Safe to call multiple times: the default handler is reset before re-adding.
    """
    if settings is None:
        from app.config import get_settings

        settings = get_settings().logging

    logger.remove()
    level = settings.level.upper()

    fmt_console = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "{message} {extra}"
    )

    logger.add(
        sys.stderr,
        level=level,
        format=fmt_console,
        serialize=settings.json_logs,
        backtrace=settings.serialize_backtrace,
        diagnose=False,  # avoid leaking variable values in tracebacks
        filter=_mask_record,
    )

    if settings.log_to_file:
        logger.add(
            settings.log_file,
            level=level,
            rotation="10 MB",
            retention="7 days",
            compression="zip",
            serialize=True,
            diagnose=False,
            filter=_mask_record,
        )

    # Bridge stdlib logging (e.g. from uvicorn/sqlalchemy) into loguru.
    class _InterceptHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            frame, depth = logging.currentframe(), 2
            while frame and frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1
            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

    logging.basicConfig(handlers=[_InterceptHandler()], level=level, force=True)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "sqlalchemy"):
        logging.getLogger(name).handlers = [_InterceptHandler()]
        logging.getLogger(name).propagate = False


def get_logger(component: str | None = None) -> Any:
    """Return a logger bound to an optional ``component`` context field."""
    if component:
        return logger.bind(component=component)
    return logger


def log_exception(
    exc: BaseException,
    *,
    component: str | None = None,
    event: str | None = None,
    task_id: str | None = None,
    agent_id: str | None = None,
    session_id: str | None = None,
) -> None:
    """Log an exception with structured context (never exposes secrets)."""
    bound = logger.bind(
        component=component,
        event=event,
        task_id=task_id,
        agent_id=agent_id,
        session_id=session_id,
        error=type(exc).__name__,
    )
    bound.exception("Exception occurred: {}", str(exc))


__all__ += ["log_exception"]

# Re-export TracebackType typing helper consumers may want.
_ExcInfo = tuple[type[BaseException], BaseException, TracebackType | None]
