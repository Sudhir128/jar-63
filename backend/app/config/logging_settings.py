"""Logging configuration."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LoggingSettings(BaseSettings):
    """Structured logging configuration."""

    level: str = Field(default="INFO", alias="LOG_LEVEL")
    json_logs: bool = Field(default=False, alias="LOG_JSON")
    log_to_file: bool = Field(default=False, alias="LOG_TO_FILE")
    log_file: str = Field(default="logs/jar63.log", alias="LOG_FILE")
    serialize_backtrace: bool = Field(default=False, alias="LOG_SERIALIZE_BACKTRACE")

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="LOG_", extra="ignore", frozen=True
    )
