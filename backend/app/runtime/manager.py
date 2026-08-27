"""Runtime manager: coordinates startup/shutdown of all subsystems.

Holds references to the registries, event bus, task/session managers,
dispatcher, workflow manager, and the loop service. Real orchestration is
deferred; this class provides the composition root that later phases extend.

Phase 3: the loop service is wired with the :class:`ToolExecutor` and policy,
and foundation tools are registered at startup (non-production).

Phase 5: the LLM subsystem (model/provider registries, router, planner, and
health checker) is wired into the runtime. The :class:`LLMPlanner` is passed
to the :class:`LoopService` so the loop uses the local Ollama model for
planning (with deterministic fallback). A startup health check runs but
never fails startup — when Ollama is unavailable the runtime degrades
gracefully.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.agents.registry import AgentRegistry
from app.config import Settings, get_settings
from app.core.logging import get_logger
from app.events import EventBus, get_event_bus
from app.llm.bootstrap import register_default_models, register_default_providers
from app.llm.health import LLMHealthChecker
from app.llm.network import HttpxNetworkChecker
from app.llm.registry import ModelRegistry, ProviderRegistry
from app.llm.router import ModelRouter
from app.memory import MemoryHealthChecker, MemoryManager, build_memory_manager, init_memory_tables
from app.runtime.dispatcher import DefaultDispatcher, Dispatcher
from app.runtime.loop.service import LoopService
from app.runtime.session_manager import SessionManager
from app.runtime.task_manager import TaskManager
from app.runtime.workflow_manager import DefaultWorkflowManager, WorkflowManager
from app.tools.impl import DEFAULT_TOOLS
from app.tools.policy import DefaultToolPolicy
from app.tools.registry import ToolRegistry

if TYPE_CHECKING:
    # LLMPlanner imports the runtime loop engine, so it is imported lazily
    # inside methods to avoid a circular import at module load time.
    from app.llm.planner import LLMPlanner

logger = get_logger("runtime")

__all__ = ["RuntimeManager", "RuntimeState"]


@dataclass
class RuntimeState:
    """Snapshot of runtime subsystem availability."""

    started: bool = False
    event_bus_ready: bool = False
    agent_registry_ready: bool = False
    tool_registry_ready: bool = False
    llm_ready: bool = False
    memory_ready: bool = False


@dataclass
class RuntimeManager:
    """Composition root for the JAR-63 runtime."""

    event_bus: EventBus = field(default_factory=get_event_bus)
    agent_registry: AgentRegistry = field(default_factory=AgentRegistry)
    tool_registry: ToolRegistry = field(default_factory=ToolRegistry)
    task_manager: TaskManager = field(default_factory=TaskManager)
    session_manager: SessionManager = field(default_factory=SessionManager)
    workflow_manager: WorkflowManager | None = None
    dispatcher: Dispatcher | None = None
    loop_service: LoopService | None = None
    # LLM subsystem (Phase 5).
    model_registry: ModelRegistry = field(default_factory=ModelRegistry)
    provider_registry: ProviderRegistry = field(default_factory=ProviderRegistry)
    model_router: ModelRouter | None = None
    llm_planner: LLMPlanner | None = None
    llm_health: LLMHealthChecker | None = None
    # Memory subsystem (Phase 6).
    memory_manager: MemoryManager | None = None
    memory_health: MemoryHealthChecker | None = None
    # Agent orchestration subsystem (Phase 7).
    agent_catalog: Any | None = None
    agent_router: Any | None = None
    agent_dispatcher: Any | None = None
    dynamic_agent_builder: Any | None = None
    agent_evaluator: Any | None = None
    lifecycle_manager: Any | None = None
    agent_orchestrator: Any | None = None
    agent_execution_store: Any | None = None
    agent_evaluation_store: Any | None = None
    settings: Settings = field(default_factory=get_settings)
    _state: RuntimeState = field(default_factory=RuntimeState)

    async def start(self) -> None:
        """Initialize subsystems and wire collaborators together."""
        if self._state.started:
            return
        self.workflow_manager = self.workflow_manager or DefaultWorkflowManager(self.event_bus)
        self.dispatcher = self.dispatcher or DefaultDispatcher(self.agent_registry, self.event_bus)

        # --- LLM subsystem (Phase 5) ---
        register_default_models(self.model_registry, self.settings)
        register_default_providers(self.provider_registry, self.settings, event_bus=self.event_bus)
        self.llm_health = LLMHealthChecker(
            provider_registry=self.provider_registry,
            model_registry=self.model_registry,
            settings=self.settings.llm,
            event_bus=self.event_bus,
        )
        # Run the startup health check. This never fails startup — when
        # Ollama is unavailable the snapshot records LOCAL_LLM_UNAVAILABLE
        # and the runtime continues with deterministic fallback.
        try:
            await self.llm_health.check()
        except Exception as exc:  # noqa: BLE001 - startup must not fail on LLM
            logger.bind(event="llm.health.startup.error", error=type(exc).__name__).warning(
                "LLM health check failed at startup (non-fatal): {}", str(exc)
            )
        self._build_router_and_planner()
        self._state.llm_ready = True

        # --- Memory subsystem (Phase 6) ---
        self._build_memory()
        await self._start_memory()

        if self.loop_service is None:
            self.loop_service = LoopService(
                task_manager=self.task_manager,
                agent_registry=self.agent_registry,
                tool_registry=self.tool_registry,
                event_bus=self.event_bus,
                session_manager=self.session_manager,
                settings=self.settings,
                tool_policy=DefaultToolPolicy(),
                llm_planner=self.llm_planner,
                memory_manager=self.memory_manager,
            )

        self._state.event_bus_ready = True
        self._state.agent_registry_ready = True
        self._state.tool_registry_ready = True
        await self._build_agent_orchestration()
        self._state.started = True
        logger.bind(
            event="runtime.started",
            llm_enabled=self.settings.llm.enabled,
            memory_enabled=self.settings.memory.enabled,
        ).info("Runtime started")

    def _build_router_and_planner(self) -> None:
        """Build the model router and LLM planner from settings."""
        if not self.settings.llm.enabled:
            self.model_router = None
            self.llm_planner = None
            return
        from app.llm.planner import LLMPlanner  # lazy import to avoid cycle

        # Cloud network probe (only used for cloud fallback; local-first
        # does not require a network check).
        cloud_probe_urls: dict[str, str] = {}
        if self.settings.llm.openai_compatible_configured:
            base = self.settings.llm.openai_compatible_base_url.rstrip("/")
            cloud_probe_urls["openai_compatible"] = f"{base}/models"
        network_checker = HttpxNetworkChecker(probe_urls=cloud_probe_urls)
        self.model_router = ModelRouter(
            model_registry=self.model_registry,
            provider_registry=self.provider_registry,
            settings=self.settings.llm,
            network_checker=network_checker,
            event_bus=self.event_bus,
        )
        self.llm_planner = LLMPlanner(
            router=self.model_router,
            settings=self.settings.llm,
            event_bus=self.event_bus,
        )

    def _build_memory(self) -> None:
        """Build the memory manager from settings (graceful when disabled).

        The manager is constructed here; ``start()`` is awaited by the caller
        (:meth:`start`) via :meth:`_start_memory`.
        """
        if not self.settings.memory.enabled:
            self.memory_manager = None
            self.memory_health = None
            return
        # Working-memory Redis client (optional; degrades without it).
        working_client = None
        if self.settings.memory.redis_working_memory:
            from app.memory.redis import get_shared_redis

            try:
                working_client = get_shared_redis()
            except Exception as exc:  # noqa: BLE001
                logger.bind(event="memory.redis.init.error", error=type(exc).__name__).warning(
                    "Redis working memory unavailable (non-fatal): {}", str(exc)
                )
        self.memory_manager = build_memory_manager(
            settings=self.settings.memory,
            event_bus=self.event_bus,
            working_client=working_client,
        )
        self.memory_health = MemoryHealthChecker(self.memory_manager)
        self._state.memory_ready = True

    async def _start_memory(self) -> None:
        """Start the memory manager (non-fatal on failure)."""
        if self.memory_manager is None:
            return
        # Ensure durable tables exist (non-fatal; PostgreSQL uses Alembic
        # in production, but create_all covers dev/test and is idempotent).
        try:
            await init_memory_tables()
        except Exception as exc:  # noqa: BLE001
            logger.bind(event="memory.tables.error", error=type(exc).__name__).warning(
                "Memory table creation failed (non-fatal): {}", str(exc)
            )
        try:
            await self.memory_manager.start()
            if self.memory_health is not None:
                await self.memory_health.check()
        except Exception as exc:  # noqa: BLE001 - startup must not fail on memory
            logger.bind(event="memory.startup.error", error=type(exc).__name__).warning(
                "Memory manager start failed (non-fatal): {}", str(exc)
            )

    async def _build_agent_orchestration(self) -> None:
        """Build the Phase 7 agent orchestration subsystem (non-fatal).

        Wires the catalog, router, dispatcher, dynamic builder, evaluator,
        lifecycle manager, and orchestrator onto the runtime. Uses the
        existing :class:`LoopService` for execution (never duplicates the loop
        engine). All LLM-dependent paths fall back to deterministic behavior
        when the LLM is unavailable. Construction never raises — if the
        subsystem cannot start, the runtime continues without it.
        """
        if not self.settings.agents.enabled:
            return
        try:
            from app.agents.catalog import AgentDefinitionRegistry
            from app.agents.dispatcher import AgentDispatcher
            from app.agents.dynamic_builder import DynamicAgentBuilder
            from app.agents.evaluation import (
                AgentEvaluator,
                DeterministicEvaluator,
                LLMEvaluator,
            )
            from app.agents.factory import default_definitions
            from app.agents.lifecycle import LifecycleManager
            from app.agents.orchestrator import AgentOrchestrator
            from app.agents.router import AgentRouter
            from app.agents.store import (
                PostgreSQLAgentDefinitionStore,
                PostgreSQLAgentEvaluationStore,
                PostgreSQLAgentExecutionStore,
                init_agent_tables,
            )
        except ImportError as exc:  # pragma: no cover - defensive
            logger.bind(event="agent.wiring.import.error", error=type(exc).__name__).warning(
                "Agent orchestration import failed (non-fatal): {}", str(exc)
            )
            return

        cfg = self.settings.agents
        # Persistence: PostgreSQL-backed stores; the async engine is
        # sqlite+aiosqlite in testing (sqlite-aware session).
        store = PostgreSQLAgentDefinitionStore()
        self.agent_execution_store = PostgreSQLAgentExecutionStore()
        self.agent_evaluation_store = PostgreSQLAgentEvaluationStore()
        self.agent_catalog = AgentDefinitionRegistry(store, event_bus=self.event_bus)
        # Create tables + seed built-in (non-dynamic) definitions (best-effort).
        try:
            await init_agent_tables()
            for d in default_definitions():
                await self.agent_catalog.upsert(d)
        except Exception as exc:  # noqa: BLE001 - seeding is best-effort
            logger.bind(event="agent.wiring.seed.error", error=type(exc).__name__).debug(
                "Agent catalog seeding skipped (non-fatal): {}", str(exc)
            )

        llm_available = self.model_router is not None
        self.agent_router = AgentRouter(
            self.agent_catalog,
            self.tool_registry,
            event_bus=self.event_bus,
            model_router=self.model_router,
            execution_store=self.agent_execution_store,
            llm_enabled=cfg.llm_routing and llm_available,
        )
        self.agent_dispatcher = AgentDispatcher(
            self.agent_catalog,
            self.agent_execution_store,
            agent_registry=self.agent_registry,
            loop_service=self.loop_service,  # type: ignore[arg-type]
            memory_manager=self.memory_manager,
            event_bus=self.event_bus,
        )
        if cfg.dynamic_agents_enabled:
            self.dynamic_agent_builder = DynamicAgentBuilder(
                self.agent_catalog,
                self.tool_registry,
                event_bus=self.event_bus,
                model_router=self.model_router,
                llm_enabled=cfg.llm_routing and llm_available,
                max_dynamic_agents=cfg.max_dynamic_agents,
                auto_activate_dynamic=cfg.auto_activate_dynamic,
            )
        strategy = (
            LLMEvaluator(model_router=self.model_router)
            if (cfg.llm_evaluation and llm_available)
            else DeterministicEvaluator()
        )
        self.agent_evaluator = AgentEvaluator(
            self.agent_execution_store,
            self.agent_evaluation_store,
            strategy=strategy,
            event_bus=self.event_bus,
            use_llm=cfg.llm_evaluation and llm_available,
        )
        self.lifecycle_manager = LifecycleManager(
            self.agent_catalog,
            self.agent_execution_store,
            evaluator=self.agent_evaluator,
            min_samples_for_retire=cfg.min_samples_for_retire,
            recent_window=cfg.recent_window,
        )
        self.agent_orchestrator = AgentOrchestrator(
            self.agent_router,
            self.agent_dispatcher,
            catalog=self.agent_catalog,
            event_bus=self.event_bus,
            dynamic_builder=self.dynamic_agent_builder,
            evaluator=self.agent_evaluator if cfg.llm_evaluation else None,
            lifecycle_manager=self.lifecycle_manager,
            memory_manager=self.memory_manager if cfg.memory_aware else None,
            allow_dynamic=cfg.dynamic_agents_enabled,
        )
        logger.bind(
            event="agent.wiring.ready",
            dynamic_agents=cfg.dynamic_agents_enabled,
            llm_routing=self.agent_router is not None and cfg.llm_routing,
        ).info("Agent orchestration subsystem ready")

    async def shutdown(self) -> None:
        """Tear down subsystems gracefully."""
        if not self._state.started:
            return
        self._state.started = False
        self._state.event_bus_ready = False
        self._state.agent_registry_ready = False
        self._state.tool_registry_ready = False
        self._state.llm_ready = False
        self._state.memory_ready = False
        # Stop the memory manager (non-fatal).
        if self.memory_manager is not None:
            with contextlib.suppress(Exception):  # noqa: BLE001
                await self.memory_manager.shutdown()
        # Close provider HTTP clients.
        for provider_id in list(self.provider_registry.list()):
            try:
                client = self.provider_registry.get(provider_id)
                await client.close()
            except Exception:  # noqa: BLE001 - shutdown must not fail
                pass
        logger.bind(event="runtime.stopped").info("Runtime stopped")

    @property
    def state(self) -> RuntimeState:
        return self._state

    @property
    def is_started(self) -> bool:
        return self._state.started

    @property
    def llm_available(self) -> bool:
        """True when the local LLM is healthy and the model is available."""
        return self.llm_health is not None and self.llm_health.snapshot.available

    @property
    def memory_available(self) -> bool:
        """True when the memory manager is enabled and started."""
        return self.memory_manager is not None and self.memory_manager.is_enabled


async def register_default_tools(registry: ToolRegistry) -> int:
    """Register the Phase 3 foundation tools + the Phase 7 search boundary.

    Foundation tools: calculator, time, health, echo (LOW risk).
    Phase 7: search (LOW risk) — the ResearchAgent research boundary.

    Returns the number of tools registered. Tools already registered are
    skipped (collision-safe).
    """
    from app.core.exceptions import ToolAlreadyRegisteredError
    from app.tools.impl import SearchTool

    tools = [*DEFAULT_TOOLS(), SearchTool()]
    registered = 0
    for tool in tools:
        try:
            await registry.register(tool)
            registered += 1
        except ToolAlreadyRegisteredError:
            logger.bind(event="bootstrap.skip", tool=tool.name).debug(
                "Tool '{}' already registered; skipping.", tool.name
            )
    logger.bind(event="bootstrap.tools", count=registered).info(
        "Registered {} tool(s).", registered
    )
    return registered
