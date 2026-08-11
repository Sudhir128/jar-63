"""Security configuration."""

from __future__ import annotations

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class SecuritySettings(BaseSettings):
    """Security configuration (CORS, secrets, token lifetimes)."""

    secret_key: SecretStr = Field(default=SecretStr("change-me-in-production"), alias="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=60, alias="ACCESS_TOKEN_EXPIRE_MINUTES")
    allowed_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:8000", "http://localhost:3000"],
        alias="ALLOWED_ORIGINS",
    )

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="SECURITY_", extra="ignore", frozen=True
    )
