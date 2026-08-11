"""API response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app import __version__

__all__ = ["HealthResponse", "VersionResponse", "ComponentStatus"]


class ComponentStatus(BaseModel):
    """Status of an individual subsystem."""

    model_config = ConfigDict(frozen=True)

    name: str
    status: str = "ok"
    detail: str | None = None


class HealthResponse(BaseModel):
    """Response model for ``GET /health``."""

    model_config = ConfigDict(frozen=True)

    status: str = "ok"
    version: str = Field(default_factory=lambda: __version__)
    components: list[ComponentStatus] = Field(default_factory=list)


class VersionResponse(BaseModel):
    """Response model for ``GET /version``."""

    model_config = ConfigDict(frozen=True)

    name: str = "JAR-63"
    version: str = Field(default_factory=lambda: __version__)
    api_version: str = "v1"
    metadata: dict[str, Any] = Field(default_factory=dict)
