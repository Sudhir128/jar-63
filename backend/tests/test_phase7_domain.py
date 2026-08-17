"""Phase 7 — agent domain models, catalog, and stores."""

from __future__ import annotations

import pytest

from app.agents.catalog import AgentDefinitionRegistry, InMemoryAgentDefinitionStore
from app.agents.definitions import (
    AgentDefinition,
    AgentEvaluation,
    AgentExecutionRecord,
    AgentKind,
    AgentLifecycleState,
    AgentPerformanceSummary,
    AgentRecommendation,
    AgentSpec,
    AgentUsageMetadata,
    AgentVersion,
)
from app.agents.interface import AgentCapability
from app.core.exceptions import (
    AgentDefinitionAlreadyExistsError,
    AgentDefinitionNotFoundError,
    AgentDefinitionValidationError,
)
from app.events import EventType, InMemoryEventBus
from app.llm.models import PrivacyLevel
from app.tools.interface import RiskLevel

# ---------------------------------------------------------------------------
# Domain models (A)
# ---------------------------------------------------------------------------


class TestAgentDefinitions:
    def _def(self, **kw) -> AgentDefinition:
        base = {
            "agent_id": "agent.x",
            "name": "X",
            "kind": AgentKind.GENERIC,
            "capabilities": {AgentCapability.REASONING},
        }
        base.update(kw)
        return AgentDefinition(**base)

    def test_defaults(self) -> None:
        d = self._def()
        assert d.lifecycle is AgentLifecycleState.ACTIVE
        assert d.is_dispatchable
        assert d.version == "0.1.0"
        assert d.privacy is PrivacyLevel.INTERNAL
        assert d.risk is RiskLevel.LOW

    def test_empty_id_rejected(self) -> None:
        with pytest.raises(ValueError):
            AgentDefinition(agent_id="", name="X")

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError):
            AgentDefinition(agent_id="a.b", name="   ")

    def test_lifecycle_dispatchable(self) -> None:
        assert AgentLifecycleState.ACTIVE.is_dispatchable
        assert not AgentLifecycleState.RETIRED.is_dispatchable
        assert not AgentLifecycleState.INACTIVE.is_dispatchable
        assert not AgentLifecycleState.DEPRECATED.is_dispatchable

    def test_usage_metadata(self) -> None:
        u = AgentUsageMetadata(dispatch_count=5, success_count=3, failure_count=2)
        assert u.dispatch_count == 5
        assert u.success_count == 3

    def test_version_validates(self) -> None:
        with pytest.raises(ValueError):
            AgentVersion(version="")

    def test_evaluation_confidence_bounds(self) -> None:
        with pytest.raises(ValueError):
            AgentEvaluation(agent_id="a", agent_version="0.1.0", confidence=1.5)

    def test_execution_record_defaults(self) -> None:
        r = AgentExecutionRecord(agent_id="a", agent_version="0.1.0", task_id="t1")
        assert r.success is False
        assert r.tool_calls == 0
        assert r.policy_violations == 0

    def test_performance_summary(self) -> None:
        s = AgentPerformanceSummary(agent_id="a", agent_version="0.1.0")
        assert s.success_rate == 0.0
        assert s.sample_count == 0

    def test_spec_validates(self) -> None:
        with pytest.raises(ValueError):
            AgentSpec(name="", purpose="p")
        with pytest.raises(ValueError):
            AgentSpec(name="n", purpose="")


# ---------------------------------------------------------------------------
# Catalog (B)
# ---------------------------------------------------------------------------


@pytest.fixture()
def catalog() -> AgentDefinitionRegistry:
    return AgentDefinitionRegistry(InMemoryAgentDefinitionStore())


@pytest.fixture()
def event_bus() -> InMemoryEventBus:
    return InMemoryEventBus()


def _math_def() -> AgentDefinition:
    return AgentDefinition(
        agent_id="math.agent",
        name="Math",
        kind=AgentKind.MATH,
        capabilities={AgentCapability.MATH},
        task_types=["math"],
        required_tools=["calculator"],
    )


class TestCatalog:
    async def test_register_and_get(self, catalog: AgentDefinitionRegistry) -> None:
        d = _math_def()
        await catalog.register(d)
        got = await catalog.get("math.agent")
        assert got.agent_id == "math.agent"
        assert (await catalog.count()) == 1

    async def test_register_duplicate_raises(self, catalog: AgentDefinitionRegistry) -> None:
        await catalog.register(_math_def())
        with pytest.raises(AgentDefinitionAlreadyExistsError):
            await catalog.register(_math_def())

    async def test_get_missing_raises(self, catalog: AgentDefinitionRegistry) -> None:
        with pytest.raises(AgentDefinitionNotFoundError):
            await catalog.get("nope")

    async def test_find_returns_none(self, catalog: AgentDefinitionRegistry) -> None:
        assert await catalog.find("nope") is None

    async def test_find_by_capability(self, catalog: AgentDefinitionRegistry) -> None:
        await catalog.register(_math_def())
        results = await catalog.find_by_capability(AgentCapability.MATH)
        assert len(results) == 1
        # inactive agents excluded by default
        await catalog.deactivate("math.agent")
        assert await catalog.find_by_capability(AgentCapability.MATH) == []
        # active_only=False includes inactive
        all_results = await catalog.find_by_capability(AgentCapability.MATH, active_only=False)
        assert len(all_results) == 1

    async def test_find_by_task_type(self, catalog: AgentDefinitionRegistry) -> None:
        await catalog.register(_math_def())
        assert len(await catalog.find_by_task_type("math")) == 1
        assert await catalog.find_by_task_type("cooking") == []

    async def test_list_active_filters(self, catalog: AgentDefinitionRegistry) -> None:
        await catalog.register(_math_def())
        await catalog.deactivate("math.agent")
        assert await catalog.list_active() == []
        assert len(await catalog.list()) == 1

    async def test_lifecycle_transitions(self, catalog: AgentDefinitionRegistry) -> None:
        await catalog.register(_math_def())
        d = await catalog.deprecate("math.agent", reason="old")
        assert d.lifecycle is AgentLifecycleState.DEPRECATED
        d = await catalog.retire("math.agent", reason="unused")
        assert d.lifecycle is AgentLifecycleState.RETIRED
        d = await catalog.activate("math.agent")
        assert d.lifecycle is AgentLifecycleState.ACTIVE

    async def test_version_creation(self, catalog: AgentDefinitionRegistry) -> None:
        await catalog.register(_math_def())
        d = await catalog.create_version("math.agent", version="0.2.0", reason="tune")
        assert d.version == "0.2.0"
        with pytest.raises(AgentDefinitionValidationError):
            await catalog.create_version("math.agent", version="0.2.0")

    async def test_record_recommendation_retire(self, catalog: AgentDefinitionRegistry) -> None:
        await catalog.register(_math_def())
        await catalog.record_recommendation("math.agent", AgentRecommendation.RETIRE, reason="eval")
        d = await catalog.get("math.agent")
        assert d.lifecycle is AgentLifecycleState.RETIRED

    async def test_record_recommendation_deactivate(self, catalog: AgentDefinitionRegistry) -> None:
        await catalog.register(_math_def())
        await catalog.record_recommendation("math.agent", AgentRecommendation.DEACTIVATE)
        d = await catalog.get("math.agent")
        assert d.lifecycle is AgentLifecycleState.INACTIVE

    async def test_record_recommendation_keep_noop(self, catalog: AgentDefinitionRegistry) -> None:
        await catalog.register(_math_def())
        await catalog.record_recommendation("math.agent", AgentRecommendation.KEEP)
        d = await catalog.get("math.agent")
        assert d.lifecycle is AgentLifecycleState.ACTIVE

    async def test_update_usage(self, catalog: AgentDefinitionRegistry) -> None:
        await catalog.register(_math_def())
        u = AgentUsageMetadata(dispatch_count=10, success_count=8)
        d = await catalog.update_usage("math.agent", u)
        assert d is not None
        assert d.usage.dispatch_count == 10

    async def test_validation_rejects_unrestricted_tools(
        self, catalog: AgentDefinitionRegistry
    ) -> None:
        bad = AgentDefinition(
            agent_id="x",
            name="X",
            required_tools=["shell"],
        )
        with pytest.raises(AgentDefinitionValidationError):
            await catalog.register(bad)

    async def test_validation_rejects_critical_dynamic(
        self, catalog: AgentDefinitionRegistry
    ) -> None:
        bad = AgentDefinition(
            agent_id="x",
            name="X",
            dynamic=True,
            risk=RiskLevel.CRITICAL,
        )
        with pytest.raises(AgentDefinitionValidationError):
            await catalog.register(bad)


# ---------------------------------------------------------------------------
# Events (N partial)
# ---------------------------------------------------------------------------


class TestCatalogEvents:
    async def test_register_emits_created(
        self, catalog: AgentDefinitionRegistry, event_bus: InMemoryEventBus
    ) -> None:
        catalog = AgentDefinitionRegistry(InMemoryAgentDefinitionStore(), event_bus=event_bus)
        events: list = []

        async def handler(ev) -> None:
            events.append(ev)

        event_bus.subscribe(EventType.AGENT_CREATED, handler)
        await catalog.register(_math_def())
        assert any(e.event_type is EventType.AGENT_CREATED for e in events)

    async def test_retire_emits_retired(
        self, catalog: AgentDefinitionRegistry, event_bus: InMemoryEventBus
    ) -> None:
        catalog = AgentDefinitionRegistry(InMemoryAgentDefinitionStore(), event_bus=event_bus)
        await catalog.register(_math_def())
        events: list = []

        async def handler(ev) -> None:
            events.append(ev)

        event_bus.subscribe(EventType.AGENT_RETIRED, handler)
        await catalog.retire("math.agent")
        assert any(e.event_type is EventType.AGENT_RETIRED for e in events)

    async def test_version_emits_version_created(
        self, catalog: AgentDefinitionRegistry, event_bus: InMemoryEventBus
    ) -> None:
        catalog = AgentDefinitionRegistry(InMemoryAgentDefinitionStore(), event_bus=event_bus)
        await catalog.register(_math_def())
        events: list = []

        async def handler(ev) -> None:
            events.append(ev)

        event_bus.subscribe(EventType.AGENT_VERSION_CREATED, handler)
        await catalog.create_version("math.agent", version="0.2.0")
        assert any(e.event_type is EventType.AGENT_VERSION_CREATED for e in events)

    async def test_events_carry_correlation(
        self, catalog: AgentDefinitionRegistry, event_bus: InMemoryEventBus
    ) -> None:
        catalog = AgentDefinitionRegistry(InMemoryAgentDefinitionStore(), event_bus=event_bus)
        events: list = []

        async def handler(ev) -> None:
            events.append(ev)

        event_bus.subscribe(None, handler)
        await catalog.register(_math_def())
        ev = events[0]
        assert ev.agent_id == "math.agent"
        assert ev.payload["version"] == "0.1.0"
        assert ev.metadata["agent_id"] == "math.agent"
        # No secrets/instructions in the event payload
        assert "instructions" not in ev.payload


# ---------------------------------------------------------------------------
# Store (DB-backed, sqlite)
# ---------------------------------------------------------------------------


class TestPostgreSQLStores:
    """Exercises the SQLAlchemy stores against the in-memory sqlite DB.

    Tests set APP_ENV=testing (conftest) so the async engine points at
    sqlite+aiosqlite:///:memory: with a shared StaticPool connection.
    """

    async def test_definition_roundtrip(self) -> None:
        from app.agents.store import (
            PostgreSQLAgentDefinitionStore,
            init_agent_tables,
        )

        await init_agent_tables()
        store = PostgreSQLAgentDefinitionStore()
        d = _math_def()
        await store.upsert(d)
        got = await store.get("math.agent")
        assert got is not None
        assert got.agent_id == "math.agent"
        assert AgentCapability.MATH in got.capabilities
        assert got.lifecycle is AgentLifecycleState.ACTIVE

    async def test_definition_lifecycle_and_usage(self) -> None:
        from app.agents.store import PostgreSQLAgentDefinitionStore, init_agent_tables

        await init_agent_tables()
        store = PostgreSQLAgentDefinitionStore()
        await store.upsert(_math_def())
        d = await store.set_lifecycle("math.agent", AgentLifecycleState.RETIRED)
        assert d.lifecycle is AgentLifecycleState.RETIRED
        d = await store.update_usage("math.agent", AgentUsageMetadata(dispatch_count=7))
        assert d.usage.dispatch_count == 7

    async def test_execution_record_store(self) -> None:
        from app.agents.store import PostgreSQLAgentExecutionStore, init_agent_tables

        await init_agent_tables()
        store = PostgreSQLAgentExecutionStore()
        # Unique ids so this test is order-independent from other test files
        # sharing the process-wide in-memory sqlite.
        agent_id = "math.agent.exec_store"
        rec = AgentExecutionRecord(
            agent_id=agent_id,
            agent_version="0.1.0",
            task_id="t1.exec_store",
            success=True,
            iterations_used=1,
            tool_calls=1,
            latency_ms=42,
        )
        await store.add(rec)
        rows = await store.list_for_agent(agent_id)
        assert len(rows) == 1
        assert rows[0].success is True
        assert rows[0].latency_ms == 42
        rows = await store.list_for_task("t1.exec_store")
        assert len(rows) == 1

    async def test_evaluation_store(self) -> None:
        from app.agents.store import PostgreSQLAgentEvaluationStore, init_agent_tables

        await init_agent_tables()
        store = PostgreSQLAgentEvaluationStore()
        agent_id = "math.agent.eval_store"
        ev = AgentEvaluation(
            agent_id=agent_id,
            agent_version="0.1.0",
            overall_assessment="reliable",
            recommendation=AgentRecommendation.KEEP,
        )
        await store.add(ev)
        rows = await store.list_for_agent(agent_id)
        assert len(rows) == 1
        assert rows[0].recommendation is AgentRecommendation.KEEP
        assert rows[0].overall_assessment == "reliable"
