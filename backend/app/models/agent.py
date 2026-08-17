"""SQLAlchemy ORM models for the agent orchestration subsystem (Phase 7).

Tables:
    * ``agent_definitions``    — persistent agent definition metadata.
    * ``agent_execution_records`` — objective execution evidence per dispatch.
    * ``agent_evaluations``    — structured agent evaluations.

These are the persistence representation; the domain layer uses the Pydantic
models in :mod:`app.agents.definitions`. Translation happens at the store
boundary (see :mod:`app.agents.store`).

Indexes cover the common retrieval paths: agent_id, lifecycle, version, and
created_at. Secrets are never stored here — definitions never carry secrets.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.core.identifiers import utc_now
from app.database import Base

__all__ = [
    "AgentDefinitionModel",
    "AgentExecutionRecordModel",
    "AgentEvaluationModel",
]


class AgentDefinitionModel(Base):
    """Persistent agent definition (PostgreSQL source of truth)."""

    __tablename__ = "agent_definitions"

    agent_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", server_default="")
    purpose: Mapped[str] = mapped_column(Text, default="", server_default="")
    kind: Mapped[str] = mapped_column(String(32), default="generic", server_default="generic")
    # Capabilities stored as a JSON array of enum values.
    capabilities: Mapped[list] = mapped_column(JSON, default=list, server_default="[]")
    task_types: Mapped[list] = mapped_column(JSON, default=list, server_default="[]")
    required_tools: Mapped[list] = mapped_column(JSON, default=list, server_default="[]")
    preferred_tools: Mapped[list] = mapped_column(JSON, default=list, server_default="[]")
    instructions: Mapped[str] = mapped_column(Text, default="", server_default="")
    constraints: Mapped[list] = mapped_column(JSON, default=list, server_default="[]")
    verification_strategy: Mapped[str] = mapped_column(Text, default="", server_default="")
    model_requirements: Mapped[list] = mapped_column(JSON, default=list, server_default="[]")
    privacy: Mapped[str] = mapped_column(String(32), default="internal", server_default="internal")
    risk: Mapped[str] = mapped_column(String(32), default="low", server_default="low")
    lifecycle: Mapped[str] = mapped_column(
        String(32), default="active", server_default="active", index=True
    )
    version: Mapped[str] = mapped_column(String(64), default="0.1.0", server_default="0.1.0")
    dynamic: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    auto_activate: Mapped[bool] = mapped_column(Boolean, default=True, server_default="1")
    usage: Mapped[dict] = mapped_column(JSON, default=dict, server_default="{}")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, index=True
    )
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, default=dict, server_default="{}")


class AgentExecutionRecordModel(Base):
    """Objective execution evidence for one agent dispatch."""

    __tablename__ = "agent_execution_records"

    record_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    agent_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    agent_version: Mapped[str] = mapped_column(String(64), nullable=False)
    task_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    loop_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    success: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    final_status: Mapped[str] = mapped_column(String(64), default="", server_default="")
    iterations_used: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    tool_calls: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    tool_failures: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    policy_violations: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    confirmations_requested: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    retries: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    cancelled: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    memory_items_used: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    model_failures: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, index=True
    )
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, default=dict, server_default="{}")

    __table_args__ = (
        Index("ix_agent_exec_agent", "agent_id", "created_at"),
        Index("ix_agent_exec_task", "task_id"),
    )


class AgentEvaluationModel(Base):
    """A structured agent evaluation (objective metrics + judgment)."""

    __tablename__ = "agent_evaluations"

    evaluation_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    agent_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    agent_version: Mapped[str] = mapped_column(String(64), nullable=False)
    objective_metrics: Mapped[dict] = mapped_column(JSON, default=dict, server_default="{}")
    overall_assessment: Mapped[str] = mapped_column(Text, default="", server_default="")
    strengths: Mapped[list] = mapped_column(JSON, default=list, server_default="[]")
    weaknesses: Mapped[list] = mapped_column(JSON, default=list, server_default="[]")
    failure_reasons: Mapped[list] = mapped_column(JSON, default=list, server_default="[]")
    capability_assessment: Mapped[dict] = mapped_column(JSON, default=dict, server_default="{}")
    recommended_changes: Mapped[list] = mapped_column(JSON, default=list, server_default="[]")
    confidence: Mapped[float] = mapped_column(Float, default=0.5, server_default="0.5")
    recommendation: Mapped[str] = mapped_column(String(32), default="keep", server_default="keep")
    evaluator: Mapped[str] = mapped_column(String(64), default="deterministic")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, index=True
    )
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, default=dict, server_default="{}")

    __table_args__ = (Index("ix_agent_eval_agent", "agent_id", "created_at"),)
