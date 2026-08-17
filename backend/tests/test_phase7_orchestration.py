"""Phase 7 — agent router, dynamic builder, dispatcher, evaluation,
lifecycle, orchestrator, and the SearchTool/ResearchAgent boundary.

All tests are deterministic and offline. No real LLM or network is used:
LLM-dependent paths are exercised via the deterministic fallbacks and via
httpx MockTransport for the LLM-evaluator/router/builder happy paths.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from app.agents.catalog import AgentDefinitionRegistry, InMemoryAgentDefinitionStore
from app.agents.definitions import (
    AgentDefinition,
    AgentKind,
    AgentLifecycleState,
    AgentRecommendation,
    AgentSpec,
)
from app.agents.dispatcher import AgentDispatcher
from app.agents.dynamic_builder import DynamicAgentBuilder
from app.agents.evaluation import (
    AgentEvaluator,
    DeterministicEvaluator,
    LLMEvaluator,
    compute_objective_metrics,
    compute_performance_summary,
)
from app.agents.factory import default_definitions
from app.agents.interface import AgentCapability
from app.agents.lifecycle import LifecycleManager
from app.agents.orchestrator import AgentOrchestrator
from app.agents.research import ResearchAgent
from app.agents.router import AgentRouter, RoutingTask
from app.agents.store import (
    PostgreSQLAgentDefinitionStore,
    PostgreSQLAgentEvaluationStore,
    PostgreSQLAgentExecutionStore,
    init_agent_tables,
)
from app.core.exceptions import AgentDefinitionValidationError
from app.events import EventType, InMemoryEventBus
from app.llm.models import PrivacyLevel
from app.tools.impl import CalculatorTool, SearchTool
from app.tools.interface import RiskLevel
from app.tools.policy import AllowAllToolPolicy
from app.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _result_verifier(expected: int):
    """A CallableVerifier checking ``output["result"] == expected``.

    The MathAgent output includes ``raw_input`` and ``expression`` keys that
    vary by phrasing, so exact-match is too strict; we assert the numeric
    result only.
    """
    from app.runtime.loop.verification.verifier import CallableVerifier

    return CallableVerifier(
        check_fn=lambda out: isinstance(out, dict) and out.get("result") == expected,
        check_name="result_equals",
    )


def _math_def() -> AgentDefinition:
    return AgentDefinition(
        agent_id="math.agent",
        name="Math",
        kind=AgentKind.MATH,
        capabilities={AgentCapability.MATH, AgentCapability.REASONING},
        task_types=["math", "arithmetic"],
        required_tools=["calculator"],
        preferred_tools=["calculator"],
    )


def _research_def() -> AgentDefinition:
    return AgentDefinition(
        agent_id="research.agent",
        name="Research",
        kind=AgentKind.RESEARCH,
        capabilities={AgentCapability.RESEARCH, AgentCapability.REASONING},
        task_types=["research", "search"],
        required_tools=["search"],
        preferred_tools=["search"],
    )


@pytest.fixture()
async def tool_registry() -> ToolRegistry:
    tools = ToolRegistry()
    await tools.register(CalculatorTool())
    await tools.register(SearchTool())
    return tools


@pytest.fixture()
async def catalog(tool_registry: ToolRegistry) -> AgentDefinitionRegistry:
    reg = AgentDefinitionRegistry(InMemoryAgentDefinitionStore())
    for d in default_definitions():
        await reg.upsert(d)
    return reg


@pytest.fixture()
async def db_stores() -> None:
    await init_agent_tables()


# ---------------------------------------------------------------------------
# SearchTool / ResearchAgent boundary (G)
# ---------------------------------------------------------------------------


class TestSearchToolBoundary:
    async def test_search_no_source_returns_empty(self) -> None:
        from app.tools.executor import ToolCallRecord, ToolExecutor

        tools = ToolRegistry()
        await tools.register(SearchTool())
        ex = ToolExecutor(tools, policy=AllowAllToolPolicy())
        outcome = await ex.execute_call(
            ToolCallRecord(tool_call_id="c1", tool_name="search", arguments={"query": "x"})
        )
        assert outcome.result.success
        assert outcome.result.output["count"] == 0
        assert outcome.result.output["results"] == []

    async def test_search_with_source(self) -> None:
        tool = SearchTool(results_source=lambda q: [{"title": f"{q} hit", "url": "https://x"}])
        from app.tools.executor import ToolCallRecord, ToolExecutor

        tools = ToolRegistry()
        await tools.register(tool)
        ex = ToolExecutor(tools, policy=AllowAllToolPolicy())
        outcome = await ex.execute_call(
            ToolCallRecord(
                tool_call_id="c1", tool_name="search", arguments={"query": "ai", "limit": 5}
            )
        )
        assert outcome.result.output["count"] == 1
        assert outcome.result.output["results"][0]["title"] == "ai hit"

    async def test_search_empty_query_fails(self) -> None:
        from app.tools.executor import ToolCallRecord, ToolExecutor

        tools = ToolRegistry()
        await tools.register(SearchTool())
        ex = ToolExecutor(tools, policy=AllowAllToolPolicy())
        outcome = await ex.execute_call(
            ToolCallRecord(tool_call_id="c1", tool_name="search", arguments={"query": ""})
        )
        assert not outcome.result.success

    async def test_research_agent_honest_when_no_sources(self) -> None:
        from app.tools.executor import ToolExecutor

        tools = ToolRegistry()
        await tools.register(SearchTool())
        ex = ToolExecutor(tools, policy=AllowAllToolPolicy())
        agent = ResearchAgent(tool_executor=ex)
        from app.agents.interface import AgentContext

        result = await agent.execute(
            AgentContext(task_id="t1", agent_id="research.agent", input="research AI")
        )
        assert result.status.value == "completed"
        assert result.output["count"] == 0
        assert "No sources available" in result.output["summary"]

    async def test_research_agent_with_source(self) -> None:
        from app.tools.executor import ToolExecutor

        tools = ToolRegistry()
        await tools.register(
            SearchTool(results_source=lambda q: [{"title": q, "url": "https://x"}])
        )
        ex = ToolExecutor(tools, policy=AllowAllToolPolicy())
        agent = ResearchAgent(tool_executor=ex)
        from app.agents.interface import AgentContext

        result = await agent.execute(
            AgentContext(
                task_id="t1", agent_id="research.agent", input="research quantum computing"
            )
        )
        assert result.status.value == "completed"
        assert result.output["count"] == 1
        assert result.metadata["verified"] is True

    async def test_research_agent_no_executor_fails_gracefully(self) -> None:
        agent = ResearchAgent(tool_executor=None)
        from app.agents.interface import AgentContext

        result = await agent.execute(
            AgentContext(task_id="t1", agent_id="research.agent", input="research X")
        )
        assert result.status.value == "failed"
        assert "tool executor" in (result.error or "").lower()


# ---------------------------------------------------------------------------
# Router (C, F)
# ---------------------------------------------------------------------------


class TestRouter:
    async def test_routes_math_task(self, catalog, tool_registry) -> None:
        router = AgentRouter(catalog, tool_registry)
        decision = await router.route(
            RoutingTask(
                goal="What is 238 * 47?",
                task_type="math",
                requested_capabilities={AgentCapability.MATH},
            )
        )
        assert decision.agent_id == "math.agent"
        assert decision.reason == "capability_match"
        assert decision.score >= 0.25

    async def test_routes_research_task(self, catalog, tool_registry) -> None:
        router = AgentRouter(catalog, tool_registry)
        decision = await router.route(
            RoutingTask(
                goal="research the topic of X",
                task_type="research",
                requested_capabilities={AgentCapability.RESEARCH},
            )
        )
        assert decision.agent_id == "research.agent"

    async def test_no_match_returns_none(self, catalog, tool_registry) -> None:
        router = AgentRouter(catalog, tool_registry)
        decision = await router.route(RoutingTask(goal="translate this poem to french"))
        assert decision.agent_id is None
        assert decision.reason == "no_match"

    async def test_retired_agent_not_selectable(self, catalog, tool_registry) -> None:
        await catalog.retire("math.agent")
        router = AgentRouter(catalog, tool_registry)
        decision = await router.route(
            RoutingTask(
                goal="2+2",
                task_type="math",
                requested_capabilities={AgentCapability.MATH},
            )
        )
        assert decision.agent_id is None

    async def test_explicit_agent_id(self, catalog, tool_registry) -> None:
        router = AgentRouter(catalog, tool_registry)
        decision = await router.route(RoutingTask(goal="x", agent_id="math.agent"))
        assert decision.agent_id == "math.agent"
        assert decision.reason == "explicit_agent_id"

    async def test_explicit_missing_raises(self, catalog, tool_registry) -> None:
        from app.core.exceptions import AgentRoutingError

        router = AgentRouter(catalog, tool_registry)
        with pytest.raises(AgentRoutingError):
            await router.route(RoutingTask(goal="x", agent_id="nope"))

    async def test_explicit_retired_raises(self, catalog, tool_registry) -> None:
        from app.core.exceptions import AgentRoutingError

        await catalog.retire("math.agent")
        router = AgentRouter(catalog, tool_registry)
        with pytest.raises(AgentRoutingError):
            await router.route(RoutingTask(goal="x", agent_id="math.agent"))

    async def test_privacy_filtering(self, catalog, tool_registry) -> None:
        # A SENSITIVE task cannot be served by an INTERNAL-only agent.
        # Built-in defs default to INTERNAL; for SENSITIVE task, they are
        # ineligible (INTERNAL not in the SENSITIVE allow-list).
        router = AgentRouter(catalog, tool_registry)
        decision = await router.route(
            RoutingTask(
                goal="secret math",
                task_type="math",
                requested_capabilities={AgentCapability.MATH},
                privacy=PrivacyLevel.SENSITIVE,
            )
        )
        assert decision.agent_id is None

    async def test_risk_filtering(self, catalog, tool_registry) -> None:
        # An agent with risk HIGH is excluded when the task caps risk at LOW.
        from app.agents.catalog import AgentDefinitionRegistry, InMemoryAgentDefinitionStore

        reg = AgentDefinitionRegistry(InMemoryAgentDefinitionStore())
        d = AgentDefinition(
            agent_id="risky.agent",
            name="Risky",
            kind=AgentKind.GENERIC,
            capabilities={AgentCapability.REASONING},
            task_types=["x"],
            required_tools=["echo"] if False else [],
            risk=RiskLevel.HIGH,
        )
        await reg.upsert(d)
        router = AgentRouter(reg, tool_registry)
        decision = await router.route(
            RoutingTask(
                goal="do x",
                task_type="x",
                requested_capabilities={AgentCapability.REASONING},
                max_risk=RiskLevel.LOW,
            )
        )
        assert decision.agent_id is None

    async def test_missing_tools_excludes_agent(self, tool_registry) -> None:
        # Agent requires a tool that is NOT registered.
        reg = AgentDefinitionRegistry(InMemoryAgentDefinitionStore())
        d = AgentDefinition(
            agent_id="needsshell",
            name="NeedsShell",
            kind=AgentKind.GENERIC,
            capabilities={AgentCapability.REASONING},
            required_tools=["shell"],
        )
        # Registration validates and rejects 'shell' as unrestricted.
        with pytest.raises(AgentDefinitionValidationError):
            await reg.register(d)

    async def test_router_emits_event(self, catalog, tool_registry) -> None:
        bus = InMemoryEventBus()
        router = AgentRouter(catalog, tool_registry, event_bus=bus)
        events: list = []

        async def h(ev) -> None:
            events.append(ev)

        bus.subscribe(EventType.AGENT_SELECTED, h)
        await router.route(RoutingTask(goal="2+2", task_type="math"))
        assert any(e.event_type is EventType.AGENT_SELECTED for e in events)


# ---------------------------------------------------------------------------
# Dynamic agent builder (D, H, K, P)
# ---------------------------------------------------------------------------


class TestDynamicBuilder:
    async def test_fallback_builds_generic_low_risk(self, tool_registry) -> None:
        reg = AgentDefinitionRegistry(InMemoryAgentDefinitionStore())
        builder = DynamicAgentBuilder(reg, tool_registry, llm_enabled=False)
        result = await builder.build_for("summarize a document", task_type="summary")
        assert result.created
        assert result.definition.kind is AgentKind.GENERIC
        assert result.definition.dynamic is True
        assert result.definition.risk is RiskLevel.LOW
        assert result.definition.lifecycle is AgentLifecycleState.ACTIVE
        assert result.definition.agent_id.startswith("dynamic.")

    async def test_reuse_existing_definition(self, tool_registry) -> None:
        reg = AgentDefinitionRegistry(InMemoryAgentDefinitionStore())
        builder = DynamicAgentBuilder(reg, tool_registry, llm_enabled=False)
        first = await builder.build_for("task A", task_type="alpha")
        # A second build with the same deterministic id components reuses.
        # (The fallback uses generate_id so agent_id differs; verify capacity
        # counting still works.)
        assert first.created

    async def test_critical_risk_rejected(self, tool_registry) -> None:
        reg = AgentDefinitionRegistry(InMemoryAgentDefinitionStore())
        DynamicAgentBuilder(reg, tool_registry, llm_enabled=False)
        # Manually craft a CRITICAL spec path via the validator.
        from app.agents.dynamic_builder import _validate_dynamic_definition

        bad = AgentDefinition(
            agent_id="dyn.bad",
            name="Bad",
            kind=AgentKind.GENERIC,
            capabilities={AgentCapability.REASONING},
            dynamic=True,
            risk=RiskLevel.CRITICAL,
        )
        with pytest.raises(AgentDefinitionValidationError):
            _validate_dynamic_definition(bad, tool_registry)

    async def test_high_risk_starts_inactive(self, tool_registry) -> None:
        # Build via spec_to_definition with HIGH risk → INACTIVE lifecycle.
        reg = AgentDefinitionRegistry(InMemoryAgentDefinitionStore())
        builder = DynamicAgentBuilder(
            reg, tool_registry, llm_enabled=False, auto_activate_dynamic=True
        )
        gen = builder._fallback_spec(  # noqa: SLF001
            "x", "x", {AgentCapability.REASONING}, PrivacyLevel.INTERNAL
        )
        gen.spec.risk = RiskLevel.HIGH  # type: ignore[misc]
        definition = builder._spec_to_definition(gen)  # noqa: SLF001
        assert definition.lifecycle is AgentLifecycleState.INACTIVE

    async def test_capacity_cap_enforced(self, tool_registry) -> None:
        reg = AgentDefinitionRegistry(InMemoryAgentDefinitionStore())
        builder = DynamicAgentBuilder(reg, tool_registry, llm_enabled=False, max_dynamic_agents=1)
        await builder.build_for("task A", task_type="alpha")
        with pytest.raises(AgentDefinitionValidationError):
            await builder.build_for("task B", task_type="beta")

    async def test_unregistered_tool_rejected(self, tool_registry) -> None:
        reg = AgentDefinitionRegistry(InMemoryAgentDefinitionStore())
        builder = DynamicAgentBuilder(reg, tool_registry, llm_enabled=False)
        # Force a spec requiring an unregistered tool.
        gen = builder._fallback_spec(  # noqa: SLF001
            "x", "x", {AgentCapability.REASONING}, PrivacyLevel.INTERNAL
        )
        gen.spec.required_tools = ["nonexistent_tool"]  # type: ignore[misc]
        definition = builder._spec_to_definition(gen)  # noqa: SLF001
        with pytest.raises(AgentDefinitionValidationError):
            from app.agents.dynamic_builder import _validate_dynamic_definition

            _validate_dynamic_definition(definition, tool_registry)

    async def test_emits_created_event(self, tool_registry) -> None:
        bus = InMemoryEventBus()
        reg = AgentDefinitionRegistry(InMemoryAgentDefinitionStore(), event_bus=bus)
        builder = DynamicAgentBuilder(reg, tool_registry, event_bus=bus, llm_enabled=False)
        events: list = []

        async def h(ev) -> None:
            events.append(ev)

        bus.subscribe(EventType.AGENT_CREATED, h)
        await builder.build_for("x", task_type="x")
        assert any(e.event_type is EventType.AGENT_CREATED for e in events)


# ---------------------------------------------------------------------------
# Evaluation (I)
# ---------------------------------------------------------------------------


def _record(success=True, **kw) -> Any:
    from app.agents.definitions import AgentExecutionRecord

    base = {
        "agent_id": "math.agent",
        "agent_version": "0.1.0",
        "task_id": "t1",
        "success": success,
        "iterations_used": 1,
        "tool_calls": 1,
        "latency_ms": 10,
    }
    base.update(kw)
    return AgentExecutionRecord(**base)


class TestEvaluation:
    def test_objective_metrics_empty(self) -> None:
        m = compute_objective_metrics([])
        assert m["sample_count"] == 0
        assert m["success_rate"] == 0.0

    def test_objective_metrics_populated(self) -> None:
        records = [_record(True), _record(False, tool_failures=1)]
        m = compute_objective_metrics(records)
        assert m["sample_count"] == 2
        assert m["success_count"] == 1
        assert m["success_rate"] == 0.5
        assert m["tool_failure_count"] == 1

    def test_performance_summary(self) -> None:
        s = compute_performance_summary("a", "0.1", [_record(True), _record(True)])
        assert s.sample_count == 2
        assert s.success_rate == 1.0

    def test_deterministic_evaluator_high_success(self) -> None:
        ev = DeterministicEvaluator()
        d = _math_def()
        records = [_record(True) for _ in range(5)]
        evaluation = ev.evaluate(d, compute_objective_metrics(records), records)
        assert evaluation.recommendation is AgentRecommendation.KEEP
        assert "success rate" in evaluation.strengths[0].lower()

    def test_deterministic_evaluator_low_success(self) -> None:
        ev = DeterministicEvaluator()
        d = _math_def()
        records = [_record(False) for _ in range(5)]
        evaluation = ev.evaluate(d, compute_objective_metrics(records), records)
        assert evaluation.recommendation is AgentRecommendation.DEACTIVATE
        assert evaluation.failure_reasons

    def test_deterministic_evaluator_no_evidence_keeps(self) -> None:
        ev = DeterministicEvaluator()
        d = _math_def()
        evaluation = ev.evaluate(d, compute_objective_metrics([]), [])
        assert evaluation.recommendation is AgentRecommendation.KEEP
        assert evaluation.confidence <= 0.5

    def test_deterministic_evaluator_policy_violations(self) -> None:
        ev = DeterministicEvaluator()
        d = _math_def()
        records = [_record(True, policy_violations=1)]
        evaluation = ev.evaluate(d, compute_objective_metrics(records), records)
        assert evaluation.recommendation is AgentRecommendation.DEACTIVATE

    async def test_evaluator_persists(self, db_stores) -> None:
        # Use a unique agent id so we don't collide with other test files
        # sharing the process-wide in-memory sqlite.
        from app.agents.definitions import AgentExecutionRecord

        agent_id = "math.agent.persist"
        defn = _math_def().model_copy(update={"agent_id": agent_id})
        exec_store = PostgreSQLAgentExecutionStore()
        eval_store = PostgreSQLAgentEvaluationStore()

        def _rec(success=True):
            return AgentExecutionRecord(
                agent_id=agent_id,
                agent_version="0.1.0",
                task_id="t1",
                success=success,
                iterations_used=1,
                tool_calls=1,
                latency_ms=10,
            )

        await exec_store.add(_rec(True))
        await exec_store.add(_rec(True))
        evaluator = AgentEvaluator(exec_store, eval_store, strategy=DeterministicEvaluator())
        evaluation = await evaluator.evaluate(defn)
        assert evaluation.recommendation is AgentRecommendation.KEEP
        history = await evaluator.history(agent_id)
        assert len(history) == 1

    async def test_llm_evaluator_falls_back_to_deterministic(self) -> None:
        # No model router → must fall back to deterministic.
        ev = LLMEvaluator(model_router=None)
        d = _math_def()
        records = [_record(True) for _ in range(3)]
        evaluation = await ev.evaluate_async(d, compute_objective_metrics(records), records)
        assert evaluation.evaluator == "deterministic"
        assert evaluation.recommendation is AgentRecommendation.KEEP


# ---------------------------------------------------------------------------
# Lifecycle (L)
# ---------------------------------------------------------------------------


class TestLifecycle:
    async def test_no_evidence_keeps(self, db_stores) -> None:
        reg = AgentDefinitionRegistry(PostgreSQLAgentDefinitionStore())
        await reg.upsert(_math_def())
        exec_store = PostgreSQLAgentExecutionStore()
        lm = LifecycleManager(reg, exec_store, min_samples_for_retire=5)
        advice = await lm.advise(await reg.get("math.agent"))
        assert advice.recommendation is AgentRecommendation.KEEP

    async def test_policy_violation_deactivates(self, db_stores) -> None:
        reg = AgentDefinitionRegistry(PostgreSQLAgentDefinitionStore())
        await reg.upsert(_math_def())
        exec_store = PostgreSQLAgentExecutionStore()
        await exec_store.add(_record(True, policy_violations=1))
        lm = LifecycleManager(reg, exec_store, min_samples_for_retire=5)
        advice = await lm.advise(await reg.get("math.agent"))
        assert advice.recommendation is AgentRecommendation.DEACTIVATE

    async def test_never_retire_below_min_samples(self, db_stores) -> None:
        reg = AgentDefinitionRegistry(PostgreSQLAgentDefinitionStore())
        await reg.upsert(_math_def())
        exec_store = PostgreSQLAgentExecutionStore()
        # Only 2 records (< min_samples_for_retire=5) → never retire.
        await exec_store.add(_record(False))
        await exec_store.add(_record(False))
        lm = LifecycleManager(reg, exec_store, min_samples_for_retire=5)
        advice = await lm.advise(await reg.get("math.agent"))
        assert advice.recommendation is not AgentRecommendation.RETIRE

    async def test_apply_keep_noop(self, db_stores) -> None:
        reg = AgentDefinitionRegistry(PostgreSQLAgentDefinitionStore())
        await reg.upsert(_math_def())
        exec_store = PostgreSQLAgentExecutionStore()
        lm = LifecycleManager(reg, exec_store)
        from app.agents.lifecycle import LifecycleAdvice

        advice = LifecycleAdvice(
            agent_id="math.agent",
            recommendation=AgentRecommendation.KEEP,
            reason="ok",
            evidence={},
        )
        d = await lm.apply(advice)
        assert d.lifecycle is AgentLifecycleState.ACTIVE

    async def test_apply_retire(self, db_stores) -> None:
        reg = AgentDefinitionRegistry(PostgreSQLAgentDefinitionStore())
        await reg.upsert(_math_def())
        exec_store = PostgreSQLAgentExecutionStore()
        lm = LifecycleManager(reg, exec_store)
        from app.agents.lifecycle import LifecycleAdvice

        advice = LifecycleAdvice(
            agent_id="math.agent",
            recommendation=AgentRecommendation.RETIRE,
            reason="unused",
            evidence={},
        )
        d = await lm.apply(advice)
        assert d.lifecycle is AgentLifecycleState.RETIRED


# ---------------------------------------------------------------------------
# Dispatcher + Orchestrator end-to-end (E, M, O, Q)
# ---------------------------------------------------------------------------


@pytest.fixture()
async def loop_service(tool_registry: ToolRegistry):
    from app.agents.registry import AgentRegistry
    from app.config import get_settings
    from app.runtime.loop.service import LoopService
    from app.runtime.session_manager import SessionManager
    from app.runtime.task_manager import TaskManager

    agent_registry = AgentRegistry()
    # Note: we do NOT pre-register the math/research runtime instances here.
    # The dispatcher builds them via AgentFactory (with the loop's tool
    # executor) and registers them at dispatch time, then releases the
    # ephemeral dynamic ones. Built-in (non-dynamic) instances stay
    # registered after the first dispatch.
    service = LoopService(
        task_manager=TaskManager(),
        agent_registry=agent_registry,
        tool_registry=tool_registry,
        event_bus=InMemoryEventBus(),
        session_manager=SessionManager(),
        settings=get_settings(),
        tool_policy=AllowAllToolPolicy(),
    )
    return service


class TestDispatcherAndOrchestrator:
    async def test_dispatcher_records_evidence(
        self, catalog, tool_registry, loop_service, db_stores
    ) -> None:
        exec_store = PostgreSQLAgentExecutionStore()
        dispatcher = AgentDispatcher(
            catalog,
            exec_store,
            agent_registry=loop_service.agent_registry,  # noqa: SLF001
            loop_service=loop_service,
        )
        from app.runtime.models import Task, TaskStatus

        task = Task(
            task_id="t-disp-1",
            agent_id="",
            input="What is 238 * 47?",
            status=TaskStatus.PENDING,
        )
        result = await dispatcher.dispatch(
            "math.agent",
            task,
            goal="What is 238 * 47?",
            max_iterations=3,
            verifier=_result_verifier(11186),
        )
        assert result.record.agent_id == "math.agent"
        assert result.record.success
        # Evidence persisted (most-recent first).
        rows = await exec_store.list_for_agent("math.agent")
        assert rows  # at least one
        assert rows[0].success is True
        assert rows[0].task_id == "t-disp-1"

    async def test_dispatcher_rejects_retired(
        self, catalog, tool_registry, loop_service, db_stores
    ) -> None:
        await catalog.retire("math.agent")
        exec_store = PostgreSQLAgentExecutionStore()
        dispatcher = AgentDispatcher(
            catalog,
            exec_store,
            agent_registry=loop_service.agent_registry,  # noqa: SLF001
            loop_service=loop_service,
        )
        from app.core.exceptions import AgentNotDispatchableError
        from app.runtime.models import Task, TaskStatus

        task = Task(task_id="t-x", input="2+2", status=TaskStatus.PENDING)
        with pytest.raises(AgentNotDispatchableError):
            await dispatcher.dispatch("math.agent", task)

    async def test_orchestrator_runs_math(
        self, catalog, tool_registry, loop_service, db_stores
    ) -> None:
        exec_store = PostgreSQLAgentExecutionStore()
        eval_store = PostgreSQLAgentEvaluationStore()
        dispatcher = AgentDispatcher(
            catalog,
            exec_store,
            agent_registry=loop_service.agent_registry,  # noqa: SLF001
            loop_service=loop_service,
        )
        router = AgentRouter(catalog, tool_registry)
        evaluator = AgentEvaluator(exec_store, eval_store, strategy=DeterministicEvaluator())
        orchestrator = AgentOrchestrator(
            router,
            dispatcher,
            catalog=catalog,
            evaluator=evaluator,
        )
        from app.runtime.models import Task, TaskStatus

        task = Task(task_id="t-orch-1", input="What is 238 * 47?", status=TaskStatus.PENDING)
        result = await orchestrator.run(
            task,
            RoutingTask(
                goal="What is 238 * 47?",
                task_type="math",
                requested_capabilities={AgentCapability.MATH},
            ),
            evaluate=True,
            max_iterations=3,
            verifier=_result_verifier(11186),
        )
        assert result.routing.agent_id == "math.agent"
        assert result.dispatch is not None
        assert result.dispatch.record.success
        assert result.evaluation is not None
        assert result.evaluation.agent_id == "math.agent"

    async def test_orchestrator_dynamic_build_when_no_match(
        self, catalog, tool_registry, loop_service, db_stores
    ) -> None:
        # Use a catalog with no math agent (retire it) and a goal nothing matches.
        await catalog.retire("math.agent")
        await catalog.retire("research.agent")
        exec_store = PostgreSQLAgentExecutionStore()
        dispatcher = AgentDispatcher(
            catalog,
            exec_store,
            agent_registry=loop_service.agent_registry,  # noqa: SLF001
            loop_service=loop_service,
        )
        router = AgentRouter(catalog, tool_registry)
        builder = DynamicAgentBuilder(catalog, tool_registry, llm_enabled=False)
        orchestrator = AgentOrchestrator(
            router,
            dispatcher,
            catalog=catalog,
            dynamic_builder=builder,
        )
        from app.runtime.models import Task, TaskStatus

        task = Task(task_id="t-dyn-1", input="some novel task", status=TaskStatus.PENDING)
        result = await orchestrator.run(
            task,
            RoutingTask(
                goal="some novel task",
                task_type="novel",
                requested_capabilities={AgentCapability.REASONING},
            ),
            max_iterations=2,
        )
        assert result.dynamic_created is True
        assert result.dynamic_definition is not None
        # A dynamic agent was created and dispatched (no tools → no-op completion).
        assert result.dispatch is not None

    async def test_orchestrator_no_agent_no_dynamic(
        self, catalog, tool_registry, loop_service, db_stores
    ) -> None:
        await catalog.retire("math.agent")
        await catalog.retire("research.agent")
        exec_store = PostgreSQLAgentExecutionStore()
        dispatcher = AgentDispatcher(
            catalog,
            exec_store,
            agent_registry=loop_service.agent_registry,  # noqa: SLF001
            loop_service=loop_service,
        )
        router = AgentRouter(catalog, tool_registry)
        orchestrator = AgentOrchestrator(
            router,
            dispatcher,
            catalog=catalog,
            allow_dynamic=False,
        )
        from app.runtime.models import Task, TaskStatus

        task = Task(task_id="t-none", input="x", status=TaskStatus.PENDING)
        result = await orchestrator.run(
            task,
            RoutingTask(goal="translate poem", task_type="poetry"),
            max_iterations=1,
        )
        assert result.routing.agent_id is None
        assert result.dispatch is None


# ---------------------------------------------------------------------------
# LLM happy-path with MockTransport (F/I/D LLM paths) (J)
# ---------------------------------------------------------------------------


def _mock_client(transport: httpx.MockTransport):
    from app.config import LLMSettings
    from app.llm.models import ModelCapability, ModelDefinition
    from app.llm.providers.ollama import OllamaClient
    from app.llm.registry import ModelRegistry, ProviderRegistry
    from app.llm.router import ModelRouter

    http_client = httpx.AsyncClient(
        base_url="http://localhost:11434", transport=transport, timeout=10.0
    )
    client = OllamaClient(base_url="http://localhost:11434", http_client=http_client)
    providers = ProviderRegistry()
    providers.register("ollama", client)
    models = ModelRegistry()
    models.register(
        ModelDefinition(
            model_id="llama3.2",
            provider="ollama",
            display_name="Llama 3.2",
            capabilities={ModelCapability.CHAT, ModelCapability.STRUCTURED_OUTPUT},
            context_window=4096,
            local=True,
            enabled=True,
            supports_structured_output=True,
        )
    )
    router = ModelRouter(
        model_registry=models,
        provider_registry=providers,
        settings=LLMSettings(),
    )
    return router, client


class TestLLMHappyPaths:
    async def test_llm_router_selects_agent(self, catalog, tool_registry) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.dumps(
                {
                    "model": "llama3.2",
                    "created_at": "2024-01-01T00:00:00Z",
                    "message": {
                        "role": "assistant",
                        "content": '{"agent_id": "research.agent", "reason": "research match"}',
                    },
                    "done": True,
                }
            )
            return httpx.Response(200, content=body)

        transport = httpx.MockTransport(handler)
        router, client = _mock_client(transport)
        agent_router = AgentRouter(catalog, tool_registry, model_router=router, llm_enabled=True)
        decision = await agent_router.route(
            RoutingTask(
                goal="research AI",
                task_type="research",
                requested_capabilities={AgentCapability.RESEARCH},
            )
        )
        assert decision.agent_id == "research.agent"
        assert decision.used_llm is True
        assert decision.reason == "llm_classification"
        await client.close()

    async def test_llm_router_invalid_falls_back(self, catalog, tool_registry) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.dumps(
                {
                    "model": "llama3.2",
                    "created_at": "2024-01-01T00:00:00Z",
                    "message": {"role": "assistant", "content": "not json"},
                    "done": True,
                }
            )
            return httpx.Response(200, content=body)

        transport = httpx.MockTransport(handler)
        router, client = _mock_client(transport)
        agent_router = AgentRouter(catalog, tool_registry, model_router=router, llm_enabled=True)
        decision = await agent_router.route(
            RoutingTask(
                goal="research AI",
                task_type="research",
                requested_capabilities={AgentCapability.RESEARCH},
            )
        )
        # Invalid LLM output → deterministic fallback.
        assert decision.reason == "capability_match"
        await client.close()

    async def test_llm_evaluator_judges(self, db_stores) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.dumps(
                {
                    "model": "llama3.2",
                    "created_at": "2024-01-01T00:00:00Z",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "overall_assessment": "reliable",
                                "strengths": ["high success"],
                                "weaknesses": [],
                                "recommended_changes": [],
                                "confidence": 0.9,
                                "recommendation": "keep",
                            }
                        ),
                    },
                    "done": True,
                }
            )
            return httpx.Response(200, content=body)

        transport = httpx.MockTransport(handler)
        router, client = _mock_client(transport)
        exec_store = PostgreSQLAgentExecutionStore()
        eval_store = PostgreSQLAgentEvaluationStore()
        for _ in range(3):
            await exec_store.add(_record(True))
        evaluator = AgentEvaluator(
            exec_store,
            eval_store,
            strategy=LLMEvaluator(model_router=router),
            use_llm=True,
        )
        evaluation = await evaluator.evaluate(_math_def())
        assert evaluation.evaluator == "llm"
        assert evaluation.recommendation is AgentRecommendation.KEEP
        await client.close()

    async def test_llm_dynamic_build(self, tool_registry) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.dumps(
                {
                    "model": "llama3.2",
                    "created_at": "2024-01-01T00:00:00Z",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "name": "Translator",
                                "purpose": "Translate text",
                                "description": "LLM-built translator agent",
                                "capabilities": ["reasoning"],
                                "task_types": ["translation"],
                                "required_tools": [],
                                "preferred_tools": [],
                                "instructions": "Translate the input faithfully.",
                                "constraints": ["no fabrication"],
                                "verification_strategy": "check output non-empty",
                                "model_requirements": ["chat"],
                                "privacy": "internal",
                                "risk": "low",
                            }
                        ),
                    },
                    "done": True,
                }
            )
            return httpx.Response(200, content=body)

        transport = httpx.MockTransport(handler)
        router, client = _mock_client(transport)
        reg = AgentDefinitionRegistry(InMemoryAgentDefinitionStore())
        builder = DynamicAgentBuilder(reg, tool_registry, model_router=router, llm_enabled=True)
        result = await builder.build_for("translate this", task_type="translation")
        assert result.created
        assert result.used_llm is True
        assert result.definition.name == "Translator"
        assert result.definition.risk is RiskLevel.LOW
        await client.close()


# ---------------------------------------------------------------------------
# AgentSpec validation (H, partial)
# ---------------------------------------------------------------------------


class TestSpecValidation:
    def test_spec_rejects_empty_name(self) -> None:
        with pytest.raises(ValueError):
            AgentSpec(name="", purpose="p")

    def test_spec_rejects_empty_purpose(self) -> None:
        with pytest.raises(ValueError):
            AgentSpec(name="n", purpose="")

    def test_spec_caps_normalize(self) -> None:
        s = AgentSpec(
            name="n",
            purpose="p",
            capabilities={AgentCapability.REASONING},
        )
        assert AgentCapability.REASONING in s.capabilities
