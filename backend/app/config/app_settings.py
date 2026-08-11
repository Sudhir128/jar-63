"""Application configuration settings."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppEnv(StrEnum):
    """Supported application environments."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class AppSettings(BaseSettings):
    """General application configuration."""

    name: str = Field(default="JAR-63", alias="APP_NAME")
    env: AppEnv = Field(default=AppEnv.DEVELOPMENT, alias="APP_ENV")
    host: str = Field(default="0.0.0.0", alias="APP_HOST")
    port: int = Field(default=8000, alias="APP_PORT")
    debug: bool = Field(default=False, alias="APP_DEBUG")
    log_level: str = Field(default="INFO", alias="APP_LOG_LEVEL")

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="APP_", extra="ignore", frozen=True
    )

    @property
    def is_dev(self) -> bool:
        return self.env is AppEnv.DEVELOPMENT

    @property
    def is_test(self) -> bool:
        return self.env is AppEnv.TESTING

    @property
    def is_prod(self) -> bool:
        return self.env is AppEnv.PRODUCTION
