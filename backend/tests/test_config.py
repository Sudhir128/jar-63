"""Tests for configuration loading and categories."""

from __future__ import annotations

from app.config import AppEnv, Settings, get_settings


def test_settings_loads_from_environment() -> None:
    settings = get_settings()
    assert isinstance(settings, Settings)
    assert settings.app.env is AppEnv.TESTING
    assert settings.app.debug is False


def test_database_settings_effective_url_prefers_explicit_url() -> None:
    settings = get_settings()
    assert settings.database.effective_url == "sqlite:///:memory:"


def test_database_settings_builds_url_from_parts(monkeypatch) -> None:
    get_settings.cache_clear()
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("POSTGRES_USER", "u")
    monkeypatch.setenv("POSTGRES_PASSWORD", "p")
    monkeypatch.setenv("POSTGRES_HOST", "h")
    monkeypatch.setenv("POSTGRES_PORT", "5433")
    monkeypatch.setenv("POSTGRES_DB", "d")
    settings = get_settings()
    assert settings.database.effective_url == "postgresql+psycopg://u:p@h:5433/d"


def test_redis_settings_effective_url() -> None:
    settings = get_settings()
    assert settings.redis.effective_url == "redis://localhost:6379/0"


def test_llm_secrets_are_masked() -> None:
    get_settings.cache_clear()
    import os

    os.environ["OPENAI_API_KEY"] = "sk-test-secret"
    try:
        settings = get_settings()
        # SecretStr does not expose the raw value via repr/str.
        assert settings.llm.openai_api_key is not None
        assert "sk-test-secret" not in repr(settings.llm.openai_api_key)
    finally:
        os.environ.pop("OPENAI_API_KEY", None)
        get_settings.cache_clear()


def test_settings_is_frozen() -> None:
    settings = get_settings()
    try:
        settings.app.name = "mutated"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("Settings should be frozen and immutable")


def test_app_env_helpers() -> None:
    settings = get_settings()
    assert settings.app.is_test is True
    assert settings.app.is_dev is False
    assert settings.app.is_prod is False
