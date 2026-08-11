"""Database (PostgreSQL) configuration."""

from __future__ import annotations

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """PostgreSQL connection configuration."""

    user: str = Field(default="jar63", alias="POSTGRES_USER")
    password: str = Field(default="jar63", alias="POSTGRES_PASSWORD")
    db: str = Field(default="jar63", alias="POSTGRES_DB")
    host: str = Field(default="postgres", alias="POSTGRES_HOST")
    port: int = Field(default=5432, alias="POSTGRES_PORT")
    url: str | None = Field(default=None, alias="DATABASE_URL")
    echo: bool = Field(default=False, alias="DATABASE_ECHO")
    pool_size: int = Field(default=10, alias="DATABASE_POOL_SIZE")
    max_overflow: int = Field(default=20, alias="DATABASE_MAX_OVERFLOW")

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="POSTGRES_", extra="ignore", frozen=True
    )

    @computed_field  # type: ignore[prop-valid]
    @property
    def effective_url(self) -> str:
        """Return the explicit URL when provided, else build from parts."""
        if self.url:
            return self.url
        return f"postgresql+psycopg://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"
