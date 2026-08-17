"""Agent domain models (Phase 7).

Persistent agent metadata: the *definition* layer. A definition describes a
kind of agent (purpose, capabilities, tools, model/privacy/risk requirements,
instructions, lifecycle state, version, usage metadata). Definitions are
deliberately separate from runtime *instances* (see :mod:`app.agents.interface`
and :class:`~app.agents.registry.AgentRegistry`).

Definitions persist; runtime instances may be ephemeral. Retiring a
definition never deletes its history — execution records and evaluations
keep referencing the agent id and version they were produced under.

Reused abstractions (no duplication):
* :class:`~app.agents.interface.AgentCapability` — the capability enum.
* :class:`~app.tools.interface.RiskLevel` — risk classification.
* :class:`~app.llm.models.PrivacyLevel` — privacy classification.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.agents.interface import AgentCapability
from app.core.identifiers import generate_id, utc_now
from app.llm.models import PrivacyLevel
from app.tools.interface import RiskLevel

__all__ = [
    "AgentLifecycleState",
    "AgentRecommendation",
    "AgentKind",
    "AgentUsageMetadata",
    "AgentVersion",
    "AgentDefinition",
    "AgentExecutionRecord",
    "AgentEvaluation",
    "AgentPerformanceSummary",
    "AgentSpec",
]


class AgentLifecycleState(StrEnum):
    """Lifecycle state of a persistent agent definition.

    Only ``ACTIVE`` definitions are selected by the router by default.
    Transitions preserve history: retiring never deletes records.
    """

    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    RETIRED = "retired"

    @property
    def is_dispatchable(self) -> bool:
        """Whether an agent in this state may be dispatched normally."""
        return self is AgentLifecycleState.ACTIVE


class AgentRecommendation(StrEnum):
    """Lifecycle recommendation produced by the evaluator/lifecycle manager."""

    KEEP = "keep"
    IMPROVE = "improve"
    DEACTIVATE = "deactivate"
    RETIRE = "retire"


class AgentKind(StrEnum):
    """How a runtime instance is constructed from a definition.

    * ``MATH``       — the existing :class:`~app.agents.math.MathAgent`.
    * ``RESEARCH``   — the :class:`~app.agents.research.ResearchAgent`.
    * ``GENERIC``    — a tool-driven agent that follows the definition's
      instructions and executes only through the :class:`ToolExecutor`.
    """

    MATH = "math"
    RESEARCH = "research"
    GENERIC = "generic"


class AgentUsageMetadata(BaseModel):
    """Roll-up usage metadata for a definition (cheap counters)."""

    model_config = ConfigDict(extra="forbid")

    dispatch_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    last_dispatched_at: datetime | None = None
    last_success_at: datetime | None = None
    last_failure_at: datetime | None = None


class AgentVersion(BaseModel):
    """A semantic version stamp for a definition.

    Execution records reference the ``version`` string they ran under, so old
    records stay interpretable after a definition is revised.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = "0.1.0"
    created_at: datetime = Field(default_factory=utc_now)
    reason: str = ""

    @field_validator("version")
    @classmethod
    def _nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("version must not be empty")
        return v


class AgentDefinition(BaseModel):
    """A persistent agent definition.

    This is the *catalog* entry the router selects from and the dispatcher
    constructs a runtime instance from. It is **not** a runnable instance.
    """

    model_config = ConfigDict(extra="forbid")

    agent_id: str
    name: str
    description: str = ""
    purpose: str = ""
    kind: AgentKind = AgentKind.GENERIC
    capabilities: set[AgentCapability] = Field(default_factory=set)
    task_types: list[str] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    preferred_tools: list[str] = Field(default_factory=list)
    instructions: str = ""
    constraints: list[str] = Field(default_factory=list)
    verification_strategy: str = ""
    model_requirements: list[str] = Field(default_factory=list)
    privacy: PrivacyLevel = PrivacyLevel.INTERNAL
    risk: RiskLevel = RiskLevel.LOW
    lifecycle: AgentLifecycleState = AgentLifecycleState.ACTIVE
    version: str = "0.1.0"
    dynamic: bool = False
    auto_activate: bool = True
    usage: AgentUsageMetadata = Field(default_factory=AgentUsageMetadata)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name", "agent_id")
    @classmethod
    def _nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty")
        return v

    @property
    def is_dispatchable(self) -> bool:
        return self.lifecycle.is_dispatchable

    def to_public_dict(self) -> dict[str, Any]:
        """API-safe dict (no secrets — definitions never carry secrets)."""
        return self.model_dump(mode="json")


class AgentExecutionRecord(BaseModel):
    """Objective evidence of one agent dispatch.

    Records only measured facts (success, iterations, latency, tool failures,
    policy violations, cancellation). It is the *evidence*, not a judgment.
    """

    model_config = ConfigDict(extra="forbid")

    record_id: str = Field(default_factory=lambda: generate_id("axrec"))
    agent_id: str
    agent_version: str
    task_id: str
    session_id: str | None = None
    loop_id: str | None = None
    success: bool = False
    final_status: str = ""
    iterations_used: int = 0
    tool_calls: int = 0
    tool_failures: int = 0
    policy_violations: int = 0
    confirmations_requested: int = 0
    retries: int = 0
    cancelled: bool = False
    latency_ms: int | None = None
    failure_reason: str | None = None
    memory_items_used: int = 0
    model_failures: int = 0
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentEvaluation(BaseModel):
    """A structured evaluation of an agent.

    Combines objective metrics (computed from evidence) with an evaluator
    judgment (LLM or deterministic strategy). The two are kept separate:
    ``objective_metrics`` are measured facts; ``assessment`` fields are
    judgments. The evaluator is a judge, never the sole source of truth.
    """

    model_config = ConfigDict(extra="forbid")

    evaluation_id: str = Field(default_factory=lambda: generate_id("aeval"))
    agent_id: str
    agent_version: str
    objective_metrics: dict[str, Any] = Field(default_factory=dict)
    overall_assessment: str = ""
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    failure_reasons: list[str] = Field(default_factory=list)
    capability_assessment: dict[str, Any] = Field(default_factory=dict)
    recommended_changes: list[str] = Field(default_factory=list)
    confidence: float = 0.5
    recommendation: AgentRecommendation = AgentRecommendation.KEEP
    evaluator: str = "deterministic"
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("confidence")
    @classmethod
    def _score(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be in [0.0, 1.0]")
        return v


class AgentPerformanceSummary(BaseModel):
    """Aggregated performance roll-up over a window of execution records."""

    model_config = ConfigDict(extra="forbid")

    agent_id: str
    agent_version: str
    sample_count: int = 0
    success_rate: float = 0.0
    avg_iterations: float = 0.0
    avg_latency_ms: float = 0.0
    tool_failure_rate: float = 0.0
    policy_violation_rate: float = 0.0
    cancellation_rate: float = 0.0
    recent_dispatch_count: int = 0
    computed_at: datetime = Field(default_factory=utc_now)


class AgentSpec(BaseModel):
    """Structured specification the :class:`DynamicAgentBuilder` validates.

    The LLM produces this; it is validated (never trusted blindly) and then
    translated into an :class:`AgentDefinition`. The LLM never writes Python
    or arbitrary executable code — only this structured description.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    purpose: str
    description: str = ""
    capabilities: list[AgentCapability] = Field(default_factory=list)
    task_types: list[str] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    preferred_tools: list[str] = Field(default_factory=list)
    instructions: str = ""
    constraints: list[str] = Field(default_factory=list)
    verification_strategy: str = ""
    model_requirements: list[str] = Field(default_factory=list)
    privacy: PrivacyLevel = PrivacyLevel.INTERNAL
    risk: RiskLevel = RiskLevel.LOW
    auto_activate: bool = True

    @field_validator("name", "purpose")
    @classmethod
    def _nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty")
        return v
