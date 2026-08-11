"""Redis configuration."""

from __future__ import annotations

from pydantic import Field, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RedisSettings(BaseSettings):
    """Redis connection configuration."""

    host: str = Field(default="redis", alias="REDIS_HOST")
    port: int = Field(default=6379, alias="REDIS_PORT")
    db: int = Field(default=0, alias="REDIS_DB")
    password: SecretStr | None = Field(default=None, alias="REDIS_PASSWORD")
    url: str | None = Field(default=None, alias="REDIS_URL")

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="REDIS_", extra="ignore", frozen=True
    )

    @computed_field  # type: ignore[prop-valid]
    @property
    def effective_url(self) -> str:
        """Return the explicit URL when provided, else build from parts."""
        if self.url:
            return self.url
        pwd = f":{self.password.get_secret_value()}@" if self.password else ""
        return f"redis://{pwd}{self.host}:{self.port}/{self.db}"
