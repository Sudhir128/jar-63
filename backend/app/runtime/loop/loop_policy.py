"""Loop policy: configurable limits enforced by the controller.

Limits are never hardcoded inside :class:`LoopController`; they are supplied
as a :class:`LoopPolicy`. The policy is the single source of truth for how
far a loop may go before it must stop.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.runtime.loop.loop_errors import LoopPolicyError

__all__ = ["LoopPolicy"]


class LoopPolicy(BaseModel):
    """Configurable execution limits for a loop."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_iterations: int = Field(default=5, ge=1, le=100)
    max_execution_time_seconds: float = Field(default=300.0, gt=0, le=3600)
    max_task_time_seconds: float = Field(default=600.0, gt=0, le=7200)
    allow_retry: bool = True
    max_retries_per_step: int = Field(default=2, ge=0, le=10)
    require_verification: bool = True
    allow_partial_completion: bool = False
    per_execution_timeout_seconds: float = Field(default=60.0, gt=0, le=600)

    # --- Phase 3: tool call limits (conservative defaults) ---
    max_tool_calls_per_iteration: int = Field(default=8, ge=1, le=50)
    max_tool_calls_per_task: int = Field(default=32, ge=1, le=200)
    max_repeated_identical_tool_calls: int = Field(default=3, ge=1, le=10)

    @model_validator(mode="after")
    def _check_bounds(self) -> LoopPolicy:
        if self.max_task_time_seconds < self.max_execution_time_seconds:
            raise LoopPolicyError("max_task_time_seconds must be >= max_execution_time_seconds")
        if not self.allow_retry and self.max_retries_per_step > 0:
            raise LoopPolicyError("max_retries_per_step must be 0 when allow_retry is False")
        if self.max_tool_calls_per_task < self.max_tool_calls_per_iteration:
            raise LoopPolicyError("max_tool_calls_per_task must be >= max_tool_calls_per_iteration")
        return self
