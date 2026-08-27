"""Tool execution configuration (Phase 8.1).

The :class:`ToolSettings` category is the **single source of truth** for
tool resource limits. The :class:`ToolExecutor` reads these at execution time
(and an explicit per-call timeout still overrides the default). Future phases
may validate result size / redirect / download / file-size limits against
these values, but nothing else may define separate duplicative defaults.
"""

from __future__ import annotations

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["ToolSettings"]


class ToolSettings(BaseSettings):
    """Tool execution resource limits.

    Environment variables (see ``.env.example``):

    * ``TOOL_MAX_EXECUTION_TIME_SECONDS`` — max wall-clock seconds a single
      tool call may run (default ``60.0``; an explicit per-call timeout wins).
    * ``TOOL_MAX_OUTPUT_SIZE_BYTES`` — soft cap on a tool's output size
      (default ``100_000``).
    * ``TOOL_MAX_REDIRECTS`` — cap on redirect hops for network tools
      (default ``5``).
    * ``TOOL_MAX_DOWNLOADED_BYTES`` — cap on bytes a network tool may download
      (default ``10_000_000``).
    * ``TOOL_MAX_FILESYSTEM_FILE_BYTES`` — cap on a single file the filesystem
      may read/write (default ``1_000_000``).
    """

    max_execution_time_seconds: float = Field(default=60.0, alias="TOOL_MAX_EXECUTION_TIME_SECONDS")
    max_output_size_bytes: int = Field(default=100_000, alias="TOOL_MAX_OUTPUT_SIZE_BYTES")
    max_redirects: int = Field(default=5, alias="TOOL_MAX_REDIRECTS")
    max_downloaded_bytes: int = Field(default=10_000_000, alias="TOOL_MAX_DOWNLOADED_BYTES")
    max_filesystem_file_bytes: int = Field(
        default=1_000_000, alias="TOOL_MAX_FILESYSTEM_FILE_BYTES"
    )

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="TOOL_", extra="ignore", frozen=True, populate_by_name=True
    )

    @model_validator(mode="after")
    def _check_limits(self) -> ToolSettings:
        if self.max_execution_time_seconds <= 0:
            raise ValueError("TOOL_MAX_EXECUTION_TIME_SECONDS must be > 0")
        for name, value in (
            ("TOOL_MAX_OUTPUT_SIZE_BYTES", self.max_output_size_bytes),
            ("TOOL_MAX_REDIRECTS", self.max_redirects),
            ("TOOL_MAX_DOWNLOADED_BYTES", self.max_downloaded_bytes),
            ("TOOL_MAX_FILESYSTEM_FILE_BYTES", self.max_filesystem_file_bytes),
        ):
            if value < 0:
                raise ValueError(f"{name} must be >= 0")
        return self
