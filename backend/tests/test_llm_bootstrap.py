"""Tests for LLM bootstrap and the 6 routing/planning demos."""

from __future__ import annotations

from app.config import LLMSettings
from app.llm.bootstrap import (
    default_model_definitions,
    register_default_models,
    register_default_providers,
)
from app.llm.demos import run_all_demos
from app.llm.providers.ollama import OLLAMA_PROVIDER_ID
from app.llm.providers.openai_compatible import OPENAI_COMPATIBLE_PROVIDER_ID
from app.llm.registry import ModelRegistry, ProviderRegistry


def _settings(*, cloud=False) -> LLMSettings:
    return LLMSettings(
        _env_file=None,
        enabled=True,
        default_provider="ollama",
        default_model="qwen2.5-coder:7b",
        ollama_base_url="http://localhost:11434",
        ollama_default_model="qwen2.5-coder:7b",
        openai_compatible_enabled=cloud,
        openai_compatible_base_url="http://cloud.example.com" if cloud else "",
        openai_compatible_model="cloud-coder:32b" if cloud else "",
    )


def test_default_model_definitions_ollama_only() -> None:
    models = default_model_definitions(_dummy_settings(_settings()))
    assert len(models) == 1
    assert models[0].provider == OLLAMA_PROVIDER_ID
    assert models[0].local is True


def test_default_model_definitions_includes_cloud_when_configured() -> None:
    models = default_model_definitions(_dummy_settings(_settings(cloud=True)))
    providers = {m.provider for m in models}
    assert OLLAMA_PROVIDER_ID in providers
    assert OPENAI_COMPATIBLE_PROVIDER_ID in providers
    cloud = [m for m in models if not m.local][0]
    assert cloud.local is False


def test_register_default_models_idempotent() -> None:
    from app.config import Settings
    from app.config.app_settings import AppEnv, AppSettings

    settings = Settings(
        app=AppSettings(env=AppEnv.TESTING),
        database=_dummy_db(),
        redis=_dummy_redis(),
        llm=_settings(),
        memory=_dummy_memory(),
        security=_dummy_sec(),
        logging=_dummy_log(),
    )
    reg = ModelRegistry()
    register_default_models(reg, settings)
    n = len(reg)
    register_default_models(reg, settings)  # second call is a no-op
    assert len(reg) == n


def test_register_default_providers() -> None:
    from app.config import Settings
    from app.config.app_settings import AppEnv, AppSettings

    settings = Settings(
        app=AppSettings(env=AppEnv.TESTING),
        database=_dummy_db(),
        redis=_dummy_redis(),
        llm=_settings(cloud=True),
        memory=_dummy_memory(),
        security=_dummy_sec(),
        logging=_dummy_log(),
    )
    reg = ProviderRegistry()
    factory = register_default_providers(reg, settings)
    assert reg.exists(OLLAMA_PROVIDER_ID)
    assert reg.exists(OPENAI_COMPATIBLE_PROVIDER_ID)
    assert factory is not None


def test_register_default_providers_no_cloud() -> None:
    from app.config import Settings
    from app.config.app_settings import AppEnv, AppSettings

    settings = Settings(
        app=AppSettings(env=AppEnv.TESTING),
        database=_dummy_db(),
        redis=_dummy_redis(),
        llm=_settings(cloud=False),
        memory=_dummy_memory(),
        security=_dummy_sec(),
        logging=_dummy_log(),
    )
    reg = ProviderRegistry()
    register_default_providers(reg, settings)
    assert reg.exists(OLLAMA_PROVIDER_ID)
    assert not reg.exists(OPENAI_COMPATIBLE_PROVIDER_ID)


def test_all_demos_pass() -> None:
    results = run_all_demos()
    assert len(results) == 6
    failed = [r for r in results if not r.passed]
    assert not failed, f"Failing demos: {[(r.name, r.detail) for r in failed]}"


# --- helpers to build a Settings without env ---


def _dummy_settings(llm: LLMSettings):
    from app.config import Settings
    from app.config.app_settings import AppEnv, AppSettings

    return Settings(
        app=AppSettings(env=AppEnv.TESTING),
        database=_dummy_db(),
        redis=_dummy_redis(),
        llm=llm,
        memory=_dummy_memory(),
        security=_dummy_sec(),
        logging=_dummy_log(),
    )


def _dummy_db():
    from app.config.database_settings import DatabaseSettings

    return DatabaseSettings()


def _dummy_redis():
    from app.config.redis_settings import RedisSettings

    return RedisSettings()


def _dummy_memory():
    from app.config.memory_settings import MemorySettings

    return MemorySettings(_env_file=None)


def _dummy_sec():
    from app.config.security_settings import SecuritySettings

    return SecuritySettings()


def _dummy_log():
    from app.config.logging_settings import LoggingSettings

    return LoggingSettings()
