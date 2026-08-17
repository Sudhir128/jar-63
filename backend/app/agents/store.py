"""Agent persistence stores (Phase 7).

Async SQLAlchemy stores for agent definitions, execution records, and
evaluations. Mirrors the memory-store pattern: the store boundary translates
between SQLAlchemy ORM rows (:mod:`app.models.agent`) and the Pydantic domain
models (:mod:`app.agents.definitions`). The domain layer never sees ORM
objects.

Works against PostgreSQL in production and SQLite in tests (both through
async SQLAlchemy). Secrets are never persisted — definitions never carry
secrets, and the store never logs content that could contain them.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from sqlalchemy import delete as sa_delete
from sqlalchemy import func, select, update

from app.agents.definitions import (
    AgentDefinition,
    AgentEvaluation,
    AgentExecutionRecord,
    AgentLifecycleState,
    AgentUsageMetadata,
)
from app.agents.interface import AgentCapability
from app.core.identifiers import utc_now
from app.core.logging import get_logger
from app.database import AsyncSessionLocal
from app.llm.models import PrivacyLevel
from app.models.agent import (
    AgentDefinitionModel,
    AgentEvaluationModel,
    AgentExecutionRecordModel,
)
from app.tools.interface import RiskLevel

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

logger = get_logger("agents.store")

__all__ = [
    "AgentDefinitionStore",
    "PostgreSQLAgentDefinitionStore",
    "AgentExecutionStore",
    "PostgreSQLAgentExecutionStore",
    "AgentEvaluationStore",
    "PostgreSQLAgentEvaluationStore",
    "init_agent_tables",
]


# ---------------------------------------------------------------------------
# Domain ⇄ ORM translation
# ---------------------------------------------------------------------------


def _row_to_definition(row: AgentDefinitionModel) -> AgentDefinition:
    caps = {AgentCapability(c) for c in (row.capabilities or []) if c}
    return AgentDefinition(
        agent_id=row.agent_id,
        name=row.name,
        description=row.description or "",
        purpose=row.purpose or "",
        kind=row.kind,  # validated by AgentDefinition
        capabilities=caps,
        task_types=list(row.task_types or []),
        required_tools=list(row.required_tools or []),
        preferred_tools=list(row.preferred_tools or []),
        instructions=row.instructions or "",
        constraints=list(row.constraints or []),
        verification_strategy=row.verification_strategy or "",
        model_requirements=list(row.model_requirements or []),
        privacy=PrivacyLevel(row.privacy),
        risk=RiskLevel(row.risk),
        lifecycle=AgentLifecycleState(row.lifecycle),
        version=row.version,
        dynamic=bool(row.dynamic),
        auto_activate=bool(row.auto_activate),
        usage=AgentUsageMetadata(**(row.usage or {})),
        created_at=row.created_at,
        updated_at=row.updated_at,
        metadata=dict(row.metadata_ or {}),
    )


def _row_to_execution(row: AgentExecutionRecordModel) -> AgentExecutionRecord:
    return AgentExecutionRecord(
        record_id=row.record_id,
        agent_id=row.agent_id,
        agent_version=row.agent_version,
        task_id=row.task_id,
        session_id=row.session_id,
        loop_id=row.loop_id,
        success=bool(row.success),
        final_status=row.final_status or "",
        iterations_used=row.iterations_used,
        tool_calls=row.tool_calls,
        tool_failures=row.tool_failures,
        policy_violations=row.policy_violations,
        confirmations_requested=row.confirmations_requested,
        retries=row.retries,
        cancelled=bool(row.cancelled),
        latency_ms=row.latency_ms,
        failure_reason=row.failure_reason,
        memory_items_used=row.memory_items_used,
        model_failures=row.model_failures,
        created_at=row.created_at,
        metadata=dict(row.metadata_ or {}),
    )


def _row_to_evaluation(row: AgentEvaluationModel) -> AgentEvaluation:
    return AgentEvaluation(
        evaluation_id=row.evaluation_id,
        agent_id=row.agent_id,
        agent_version=row.agent_version,
        objective_metrics=dict(row.objective_metrics or {}),
        overall_assessment=row.overall_assessment or "",
        strengths=list(row.strengths or []),
        weaknesses=list(row.weaknesses or []),
        failure_reasons=list(row.failure_reasons or []),
        capability_assessment=dict(row.capability_assessment or {}),
        recommended_changes=list(row.recommended_changes or []),
        confidence=float(row.confidence),
        recommendation=row.recommendation,  # validated by AgentEvaluation
        evaluator=row.evaluator or "deterministic",
        created_at=row.created_at,
        metadata=dict(row.metadata_ or {}),
    )


# ---------------------------------------------------------------------------
# Definition store
# ---------------------------------------------------------------------------


class AgentDefinitionStore(abc.ABC):
    """Persistence contract for agent definitions."""

    @abc.abstractmethod
    async def upsert(self, definition: AgentDefinition) -> AgentDefinition:
        """Insert or replace a definition (keyed by agent_id)."""

    @abc.abstractmethod
    async def get(self, agent_id: str) -> AgentDefinition | None:
        """Return a definition by id, or None."""

    @abc.abstractmethod
    async def list(
        self,
        *,
        lifecycle: AgentLifecycleState | None = None,
        limit: int = 200,
    ) -> list[AgentDefinition]:
        """List definitions, optionally filtered by lifecycle."""

    @abc.abstractmethod
    async def list_active(self, limit: int = 200) -> list[AgentDefinition]:
        """List active definitions only."""

    @abc.abstractmethod
    async def set_lifecycle(
        self, agent_id: str, lifecycle: AgentLifecycleState
    ) -> AgentDefinition | None:
        """Transition a definition's lifecycle state."""

    @abc.abstractmethod
    async def update_usage(
        self, agent_id: str, usage: AgentUsageMetadata
    ) -> AgentDefinition | None:
        """Update the roll-up usage metadata for a definition."""

    @abc.abstractmethod
    async def update(self, agent_id: str, **fields: object) -> AgentDefinition | None:
        """Update arbitrary fields on a definition."""

    @abc.abstractmethod
    async def delete(self, agent_id: str) -> bool:
        """Hard-delete a definition (history preserved elsewhere)."""

    @abc.abstractmethod
    async def count(self) -> int:
        """Count all definitions."""


class PostgreSQLAgentDefinitionStore(AgentDefinitionStore):
    """PostgreSQL/SQLite-backed definition store."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession] | None = None) -> None:
        self._sf = session_factory or AsyncSessionLocal

    async def upsert(self, definition: AgentDefinition) -> AgentDefinition:
        async with self._sf() as session:
            existing = await session.get(AgentDefinitionModel, definition.agent_id)
            row = _definition_to_row(definition, existing)
            if existing is None:
                session.add(row)
            await session.commit()
            return definition

    async def get(self, agent_id: str) -> AgentDefinition | None:
        async with self._sf() as session:
            row = await session.get(AgentDefinitionModel, agent_id)
            return _row_to_definition(row) if row else None

    async def list(
        self,
        *,
        lifecycle: AgentLifecycleState | None = None,
        limit: int = 200,
    ) -> list[AgentDefinition]:
        stmt = select(AgentDefinitionModel)
        if lifecycle is not None:
            stmt = stmt.where(AgentDefinitionModel.lifecycle == lifecycle.value)
        stmt = stmt.order_by(AgentDefinitionModel.created_at.desc()).limit(limit)
        async with self._sf() as session:
            rows = (await session.execute(stmt)).scalars().all()
            return [_row_to_definition(r) for r in rows]

    async def list_active(self, limit: int = 200) -> list[AgentDefinition]:
        return await self.list(lifecycle=AgentLifecycleState.ACTIVE, limit=limit)

    async def set_lifecycle(
        self, agent_id: str, lifecycle: AgentLifecycleState
    ) -> AgentDefinition | None:
        async with self._sf() as session:
            row = await session.get(AgentDefinitionModel, agent_id)
            if row is None:
                return None
            await session.execute(
                update(AgentDefinitionModel)
                .where(AgentDefinitionModel.agent_id == agent_id)
                .values(lifecycle=lifecycle.value, updated_at=utc_now())
            )
            await session.commit()
            row = await session.get(AgentDefinitionModel, agent_id)
            return _row_to_definition(row) if row else None

    async def update_usage(
        self, agent_id: str, usage: AgentUsageMetadata
    ) -> AgentDefinition | None:
        async with self._sf() as session:
            row = await session.get(AgentDefinitionModel, agent_id)
            if row is None:
                return None
            await session.execute(
                update(AgentDefinitionModel)
                .where(AgentDefinitionModel.agent_id == agent_id)
                .values(usage=usage.model_dump(mode="json"), updated_at=utc_now())
            )
            await session.commit()
            row = await session.get(AgentDefinitionModel, agent_id)
            return _row_to_definition(row) if row else None

    async def update(self, agent_id: str, **fields: object) -> AgentDefinition | None:
        allowed = {
            "name",
            "description",
            "purpose",
            "instructions",
            "constraints",
            "verification_strategy",
            "required_tools",
            "preferred_tools",
            "task_types",
            "model_requirements",
            "privacy",
            "risk",
            "version",
            "auto_activate",
            "metadata",
        }
        clean: dict[str, object] = {}
        for k, v in fields.items():
            if k in allowed:
                clean[k if k != "metadata" else "metadata_"] = v
        if not clean:
            return await self.get(agent_id)
        clean["updated_at"] = utc_now()
        async with self._sf() as session:
            row = await session.get(AgentDefinitionModel, agent_id)
            if row is None:
                return None
            await session.execute(
                update(AgentDefinitionModel)
                .where(AgentDefinitionModel.agent_id == agent_id)
                .values(**clean)
            )
            await session.commit()
            row = await session.get(AgentDefinitionModel, agent_id)
            return _row_to_definition(row) if row else None

    async def delete(self, agent_id: str) -> bool:
        async with self._sf() as session:
            result = await session.execute(
                sa_delete(AgentDefinitionModel).where(AgentDefinitionModel.agent_id == agent_id)
            )
            await session.commit()
            return (result.rowcount or 0) > 0

    async def count(self) -> int:
        async with self._sf() as session:
            result = await session.execute(select(func.count()).select_from(AgentDefinitionModel))
            return int(result.scalar() or 0)


def _definition_to_row(
    definition: AgentDefinition, existing: AgentDefinitionModel | None
) -> AgentDefinitionModel:
    row = existing or AgentDefinitionModel(agent_id=definition.agent_id)
    row.name = definition.name
    row.description = definition.description
    row.purpose = definition.purpose
    row.kind = definition.kind.value
    row.capabilities = [c.value for c in definition.capabilities]
    row.task_types = list(definition.task_types)
    row.required_tools = list(definition.required_tools)
    row.preferred_tools = list(definition.preferred_tools)
    row.instructions = definition.instructions
    row.constraints = list(definition.constraints)
    row.verification_strategy = definition.verification_strategy
    row.model_requirements = list(definition.model_requirements)
    row.privacy = definition.privacy.value
    row.risk = definition.risk.value
    row.lifecycle = definition.lifecycle.value
    row.version = definition.version
    row.dynamic = definition.dynamic
    row.auto_activate = definition.auto_activate
    row.usage = definition.usage.model_dump(mode="json")
    row.created_at = definition.created_at
    row.updated_at = definition.updated_at
    row.metadata_ = dict(definition.metadata)
    return row


# ---------------------------------------------------------------------------
# Execution record store
# ---------------------------------------------------------------------------


class AgentExecutionStore(abc.ABC):
    """Persistence contract for objective execution evidence."""

    @abc.abstractmethod
    async def add(self, record: AgentExecutionRecord) -> AgentExecutionRecord:
        """Insert an execution record."""

    @abc.abstractmethod
    async def list_for_agent(
        self, agent_id: str, *, limit: int = 100
    ) -> list[AgentExecutionRecord]:
        """List execution records for an agent (newest first)."""

    @abc.abstractmethod
    async def list_for_task(self, task_id: str) -> list[AgentExecutionRecord]:
        """List execution records for a task."""

    @abc.abstractmethod
    async def count(self) -> int:
        """Count all records."""


class PostgreSQLAgentExecutionStore(AgentExecutionStore):
    """PostgreSQL/SQLite-backed execution record store."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession] | None = None) -> None:
        self._sf = session_factory or AsyncSessionLocal

    async def add(self, record: AgentExecutionRecord) -> AgentExecutionRecord:
        async with self._sf() as session:
            row = AgentExecutionRecordModel(
                record_id=record.record_id,
                agent_id=record.agent_id,
                agent_version=record.agent_version,
                task_id=record.task_id,
                session_id=record.session_id,
                loop_id=record.loop_id,
                success=record.success,
                final_status=record.final_status,
                iterations_used=record.iterations_used,
                tool_calls=record.tool_calls,
                tool_failures=record.tool_failures,
                policy_violations=record.policy_violations,
                confirmations_requested=record.confirmations_requested,
                retries=record.retries,
                cancelled=record.cancelled,
                latency_ms=record.latency_ms,
                failure_reason=record.failure_reason,
                memory_items_used=record.memory_items_used,
                model_failures=record.model_failures,
                created_at=record.created_at,
                metadata_=dict(record.metadata),
            )
            session.add(row)
            await session.commit()
            return record

    async def list_for_agent(
        self, agent_id: str, *, limit: int = 100
    ) -> list[AgentExecutionRecord]:
        stmt = (
            select(AgentExecutionRecordModel)
            .where(AgentExecutionRecordModel.agent_id == agent_id)
            .order_by(AgentExecutionRecordModel.created_at.desc())
            .limit(limit)
        )
        async with self._sf() as session:
            rows = (await session.execute(stmt)).scalars().all()
            return [_row_to_execution(r) for r in rows]

    async def list_for_task(self, task_id: str) -> list[AgentExecutionRecord]:
        stmt = (
            select(AgentExecutionRecordModel)
            .where(AgentExecutionRecordModel.task_id == task_id)
            .order_by(AgentExecutionRecordModel.created_at.desc())
        )
        async with self._sf() as session:
            rows = (await session.execute(stmt)).scalars().all()
            return [_row_to_execution(r) for r in rows]

    async def count(self) -> int:
        async with self._sf() as session:
            result = await session.execute(
                select(func.count()).select_from(AgentExecutionRecordModel)
            )
            return int(result.scalar() or 0)


# ---------------------------------------------------------------------------
# Evaluation store
# ---------------------------------------------------------------------------


class AgentEvaluationStore(abc.ABC):
    """Persistence contract for agent evaluations."""

    @abc.abstractmethod
    async def add(self, evaluation: AgentEvaluation) -> AgentEvaluation:
        """Insert an evaluation."""

    @abc.abstractmethod
    async def list_for_agent(self, agent_id: str, *, limit: int = 50) -> list[AgentEvaluation]:
        """List evaluations for an agent (newest first)."""

    @abc.abstractmethod
    async def count(self) -> int:
        """Count all evaluations."""


class PostgreSQLAgentEvaluationStore(AgentEvaluationStore):
    """PostgreSQL/SQLite-backed evaluation store."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession] | None = None) -> None:
        self._sf = session_factory or AsyncSessionLocal

    async def add(self, evaluation: AgentEvaluation) -> AgentEvaluation:
        async with self._sf() as session:
            row = AgentEvaluationModel(
                evaluation_id=evaluation.evaluation_id,
                agent_id=evaluation.agent_id,
                agent_version=evaluation.agent_version,
                objective_metrics=dict(evaluation.objective_metrics),
                overall_assessment=evaluation.overall_assessment,
                strengths=list(evaluation.strengths),
                weaknesses=list(evaluation.weaknesses),
                failure_reasons=list(evaluation.failure_reasons),
                capability_assessment=dict(evaluation.capability_assessment),
                recommended_changes=list(evaluation.recommended_changes),
                confidence=evaluation.confidence,
                recommendation=evaluation.recommendation.value,
                evaluator=evaluation.evaluator,
                created_at=evaluation.created_at,
                metadata_=dict(evaluation.metadata),
            )
            session.add(row)
            await session.commit()
            return evaluation

    async def list_for_agent(self, agent_id: str, *, limit: int = 50) -> list[AgentEvaluation]:
        stmt = (
            select(AgentEvaluationModel)
            .where(AgentEvaluationModel.agent_id == agent_id)
            .order_by(AgentEvaluationModel.created_at.desc())
            .limit(limit)
        )
        async with self._sf() as session:
            rows = (await session.execute(stmt)).scalars().all()
            return [_row_to_evaluation(r) for r in rows]

    async def count(self) -> int:
        async with self._sf() as session:
            result = await session.execute(select(func.count()).select_from(AgentEvaluationModel))
            return int(result.scalar() or 0)


async def init_agent_tables() -> None:
    """Create agent tables if they do not exist (idempotent)."""
    import app.models.agent  # noqa: F401
    from app.database import Base, async_engine

    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
