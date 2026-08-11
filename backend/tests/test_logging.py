"""Tests for structured logging and secret masking."""

from __future__ import annotations

from io import StringIO

from loguru import logger

from app.core.logging import _mask_value


def test_mask_value_redacts_sensitive_keys() -> None:
    assert _mask_value("password", "hunter2") == "<redacted>"
    assert _mask_value("api_key", "sk-xxx") == "<redacted>"
    assert _mask_value("token", "abc") == "<redacted>"
    assert _mask_value("authorization", "Bearer x") == "<redacted>"


def test_mask_value_preserves_non_sensitive() -> None:
    assert _mask_value("name", "JAR-63") == "JAR-63"
    assert _mask_value("count", 5) == 5
    assert _mask_value("none", None) is None


def test_mask_value_redacts_url_passwords() -> None:
    masked = _mask_value("database_url", "postgresql://u:secret@host:5432/db")
    assert "secret" not in masked
    assert "<redacted>" in masked


def test_mask_value_redacts_bearer_tokens() -> None:
    masked = _mask_value("header", "Bearer abc.def.ghi")
    assert "abc.def.ghi" not in masked
    assert "<redacted>" in masked


def test_logger_binds_component() -> None:
    sink = StringIO()
    handler_id = logger.add(sink, format="{message} {extra}", level="INFO")
    try:
        logger.bind(component="api").info("hello")
        contents = sink.getvalue()
        assert "hello" in contents
        # loguru serializes the extra dict inline.
        assert "'component': 'api'" in contents
    finally:
        logger.remove(handler_id)
