"""Tests for the extended LLM configuration."""

from __future__ import annotations

import os

from app.config import get_settings
from app.config.llm_settings import LLMSettings


def test_llm_settings_defaults_local_first() -> None:
    # Use _env_file=None so the test is isolated from any .env file present
    # in the working directory (e.g. inside the Docker container).
    s = LLMSettings(_env_file=None)
    assert s.enabled is True
    assert s.default_provider == "ollama"
    assert s.default_model == "qwen2.5-coder:7b"
    assert s.routing_policy == "local_first"
    assert s.ollama_base_url == "http://localhost:11434"
    assert s.openai_compatible_enabled is False
    assert s.allow_cloud_for_private is False


def test_llm_settings_no_openai_key_required() -> None:
    s = LLMSettings(_env_file=None)
    # No API key is required to construct settings. When the env var is set
    # to an empty string, pydantic returns SecretStr(''); either way, no
    # usable secret is present.
    assert (
        s.openai_compatible_api_key is None or s.openai_compatible_api_key.get_secret_value() == ""
    )


def test_openai_compatible_not_configured_when_disabled() -> None:
    s = LLMSettings(_env_file=None)
    assert s.openai_compatible_configured is False


def test_openai_compatible_configured_when_enabled() -> None:
    s = LLMSettings(
        _env_file=None,
        openai_compatible_enabled=True,
        openai_compatible_base_url="http://cloud.example.com",
        openai_compatible_model="cloud-coder:32b",
    )
    assert s.openai_compatible_configured is True


def test_openai_compatible_api_key_is_secret_str() -> None:
    from pydantic import SecretStr

    s = LLMSettings(_env_file=None, openai_compatible_api_key="sk-test")
    assert isinstance(s.openai_compatible_api_key, SecretStr)
    assert "sk-test" not in repr(s.openai_compatible_api_key)


def test_settings_load_from_environment(monkeypatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("LLM_ENABLED", "false")
    monkeypatch.setenv("LLM_ROUTING_POLICY", "local_only")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    monkeypatch.setenv("OLLAMA_DEFAULT_MODEL", "llama3.2:3b")
    try:
        s = LLMSettings(_env_file=None)
        assert s.enabled is False
        assert s.routing_policy == "local_only"
        assert s.ollama_base_url == "http://host.docker.internal:11434"
        assert s.ollama_default_model == "llama3.2:3b"
    finally:
        get_settings.cache_clear()


def test_legacy_openai_key_still_supported(monkeypatch) -> None:
    """Backward compatibility: the existing OPENAI_API_KEY field is retained."""
    get_settings.cache_clear()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-legacy")
    try:
        s = LLMSettings(_env_file=None)
        assert s.openai_api_key is not None
        assert "sk-legacy" not in repr(s.openai_api_key)
    finally:
        os.environ.pop("OPENAI_API_KEY", None)
        get_settings.cache_clear()
