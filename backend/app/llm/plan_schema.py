"""Strict LLM plan schema.

The LLM planner produces a :class:`LLMPlan` conforming to this schema. It is
validated before being translated into the loop engine's
:class:`~app.runtime.loop.stages.PlanResult` / :class:`NextAction`.

The LLM **never** executes anything — it only describes a plan. Execution
remains the responsibility of the Loop Engine, the Agent Registry, and the
Tool Registry.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.core.identifiers import generate_id
from app.llm.models import ModelCapability

__all__ = ["LLMPlanStep", "LLMPlan"]


class LLMPlanStep(BaseModel):
    """A single step in an LLM-generated plan.

    ``capability`` declares what kind of work the step needs; it is matched
    against model/agent capabilities, never executed directly.

    When ``tool`` is set, the plan requests execution of a registered tool
    with the given ``tool_arguments``. The tool is resolved through the
    :class:`~app.tools.registry.ToolRegistry` and must pass policy — the LLM
    never executes tools directly.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    step_id: str = Field(default_factory=lambda: generate_id("pstep"))
    description: str
    capability: str = ModelCapability.CHAT.value
    agent_id: str | None = None
    tool: str | None = None
    tool_arguments: dict[str, Any] = Field(default_factory=dict)
    tool_requirements: list[str] = Field(default_factory=list)
    expected_output: str = ""
    verification_requirements: list[str] = Field(default_factory=list)

    @field_validator("description")
    @classmethod
    def _description_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("step description must not be empty")
        return v

    @property
    def is_tool_step(self) -> bool:
        return self.tool is not None


class LLMPlan(BaseModel):
    """A validated structured plan produced by the LLM planner."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    plan_id: str = Field(default_factory=lambda: generate_id("llmplan"))
    goal: str
    steps: list[LLMPlanStep] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)

    @field_validator("goal")
    @classmethod
    def _goal_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("plan goal must not be empty")
        return v

    @model_validator(mode="after")
    def _require_success_criteria(self) -> LLMPlan:
        if not self.success_criteria:
            raise ValueError("plan must include at least one success criterion")
        return self

    @property
    def first_step(self) -> LLMPlanStep | None:
        return self.steps[0] if self.steps else None
