"""Centralized configuration for JAR-63.

All settings are loaded from environment variables (optionally a `.env` file).
Configuration is frozen at load time and organized into categories.

Each category is an independent ``BaseSettings`` that reads its own env vars,
so they can also be used standalone. The aggregate :class:`Settings` is a
plain frozen model that composes them; :func:`get_settings` builds each
category from the environment and wires them together.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import BaseModel, ConfigDict, Field

from app.config.agent_settings import AgentSettings
from app.config.app_settings import AppEnv, AppSettings
from app.config.database_settings import DatabaseSettings
from app.config.llm_settings import LLMSettings
from app.config.logging_settings import LoggingSettings
from app.config.memory_settings import MemorySettings
from app.config.redis_settings import RedisSettings
from app.config.security_settings import SecuritySettings

__all__ = [
    "AgentSettings",
    "AppEnv",
    "AppSettings",
    "DatabaseSettings",
    "LLMSettings",
    "LoggingSettings",
    "MemorySettings",
    "RedisSettings",
    "SecuritySettings",
    "Settings",
    "get_settings",
]


class Settings(BaseModel):
    """Aggregate of all configuration categories."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    app: AppSettings = Field(default_factory=AppSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    agents: AgentSettings = Field(default_factory=AgentSettings)

    @property
    def is_testing(self) -> bool:
        return self.app.env is AppEnv.TESTING


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached, immutable :class:`Settings` instance."""
    return Settings(
        app=AppSettings(),
        database=DatabaseSettings(),
        redis=RedisSettings(),
        llm=LLMSettings(),
        memory=MemorySettings(),
        security=SecuritySettings(),
        logging=LoggingSettings(),
        agents=AgentSettings(),
    )
