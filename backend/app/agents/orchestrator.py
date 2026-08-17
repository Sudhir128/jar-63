"""Agent orchestrator (Phase 7).

Ties the Phase 7 components together:

    RoutingTask
        ↓
    AgentRouter            ── no match ──→ DynamicAgentBuilder
        ↓ (RoutingDecision)
    AgentDispatcher        ── executes via the existing Loop Engine
        ↓ (AgentExecutionRecord)
    (optional) AgentEvaluator ── AgentEvaluation
        ↓
    (optional) LifecycleManager ── LifecycleAdvice / transition
        ↓
    (optional) memory persistence of the execution

The orchestrator is the single entry point for "run this goal as an agent". It
degrades gracefully: when the LLM is unavailable, routing and dynamic building
fall back to deterministic behavior. It never duplicates the loop engine —
execution always goes through :class:`LoopService`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.agents.dispatcher import DispatchResult
from app.agents.router import RoutingDecision, RoutingTask
from app.core.logging import get_logger
from app.events import EventBus
from app.runtime.models import Task

if TYPE_CHECKING:
    from app.agents.catalog import AgentDefinitionRegistry
    from app.agents.dispatcher import AgentDispatcher
    from app.agents.dynamic_builder import DynamicAgentBuilder
    from app.agents.evaluation import AgentEvaluator
    from app.agents.lifecycle import LifecycleManager
    from app.agents.router import AgentRouter
    from app.memory.manager import MemoryManager

logger = get_logger("agents.orchestrator")

__all__ = ["AgentOrchestrator", "OrchestrationResult"]


@dataclass
class OrchestrationResult:
    """The complete result of an orchestration run."""

    routing: RoutingDecision
    dispatch: DispatchResult | None
    evaluation: Any | None = None
    lifecycle_advice: Any | None = None
    dynamic_created: bool = False
    dynamic_definition: Any | None = None
    memory_record_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentOrchestrator:
    """Coordinates routing → dynamic build → dispatch → evaluation → memory."""

    def __init__(
        self,
        router: AgentRouter,
        dispatcher: AgentDispatcher,
        *,
        catalog: AgentDefinitionRegistry,
        event_bus: EventBus | None = None,
        dynamic_builder: DynamicAgentBuilder | None = None,
        evaluator: AgentEvaluator | None = None,
        lifecycle_manager: LifecycleManager | None = None,
        memory_manager: MemoryManager | None = None,
        allow_dynamic: bool = True,
    ) -> None:
        self._router = router
        self._dispatcher = dispatcher
        self._catalog = catalog
        self._event_bus = event_bus
        self._builder = dynamic_builder
        self._evaluator = evaluator
        self._lifecycle = lifecycle_manager
        self._memory = memory_manager
        self._allow_dynamic = allow_dynamic

    async def run(
        self,
        task: Task,
        routing: RoutingTask,
        *,
        evaluate: bool = False,
        apply_lifecycle: bool = False,
        persist_memory: bool = False,
        max_iterations: int = 5,
        success_criteria: list[str] | None = None,
        expected_output: Any = None,
        verifier: Any | None = None,
    ) -> OrchestrationResult:
        """Run a full agent orchestration for a task + routing request."""
        decision = await self._router.route(routing)
        if decision.agent_id is None:
            if self._allow_dynamic and self._builder is not None:
                created = await self._maybe_build_dynamic(routing)
                if created is not None:
                    decision = RoutingDecision(
                        agent_id=created.definition.agent_id,
                        reason="dynamic_agent_created",
                        candidates_considered=0,
                        used_llm=False,
                        fallback=True,
                        metadata={"dynamic": True},
                    )
                    result = OrchestrationResult(
                        routing=decision,
                        dispatch=None,
                        dynamic_created=created.created,
                        dynamic_definition=created.definition,
                    )
                    # Dispatch the freshly-built agent immediately.
                    dispatch = await self._dispatch(
                        decision,
                        task,
                        routing,
                        max_iterations,
                        success_criteria,
                        expected_output,
                        verifier,
                    )
                    result.dispatch = dispatch
                    if evaluate and dispatch is not None and self._evaluator is not None:
                        result.evaluation = await self._evaluate(dispatch)
                    return result
            # No agent and no dynamic build → return without dispatching.
            return OrchestrationResult(routing=decision, dispatch=None)

        dispatch = await self._dispatch(
            decision,
            task,
            routing,
            max_iterations,
            success_criteria,
            expected_output,
            verifier,
        )
        result = OrchestrationResult(routing=decision, dispatch=dispatch)
        if evaluate and dispatch is not None and self._evaluator is not None:
            result.evaluation = await self._evaluate(dispatch)
        if apply_lifecycle and dispatch is not None and self._lifecycle is not None:
            result.lifecycle_advice = await self._lifecycle.advise(dispatch.definition)
        if persist_memory and dispatch is not None and self._memory is not None:
            result.memory_record_id = await self._persist_memory(dispatch)
        return result

    # --- helpers --------------------------------------------------------

    async def _maybe_build_dynamic(self, routing: RoutingTask):
        try:
            return await self._builder.build_for(
                routing.goal,
                task_type=routing.task_type,
                requested_capabilities=routing.requested_capabilities,
                privacy=routing.privacy,
            )
        except Exception as exc:  # noqa: BLE001 - dynamic build failure is non-fatal
            logger.bind(
                event="agent.orchestration.dynamic.error", error=type(exc).__name__
            ).warning("Dynamic agent build failed: {}", str(exc))
            return None

    async def _dispatch(
        self,
        decision: RoutingDecision,
        task: Task,
        routing: RoutingTask,
        max_iterations: int,
        success_criteria: list[str] | None,
        expected_output: Any,
        verifier: Any | None,
    ) -> DispatchResult | None:
        try:
            return await self._dispatcher.dispatch(
                decision.agent_id,  # type: ignore[arg-type]
                task,
                goal=routing.goal,
                success_criteria=success_criteria,
                max_iterations=max_iterations,
                expected_output=expected_output,
                verifier=verifier,
                session_id=routing.metadata.get("session_id"),
            )
        except Exception as exc:  # noqa: BLE001 - never fabricate a dispatch
            logger.bind(
                event="agent.orchestration.dispatch.error", error=type(exc).__name__
            ).warning("Dispatch failed: {}", str(exc))
            return None

    async def _evaluate(self, dispatch: DispatchResult) -> Any:
        try:
            return await self._evaluator.evaluate(dispatch.definition)
        except Exception as exc:  # noqa: BLE001
            logger.bind(event="agent.orchestration.eval.error", error=type(exc).__name__).warning(
                "Evaluation failed: {}", str(exc)
            )
            return None

    async def _persist_memory(self, dispatch: DispatchResult) -> str | None:
        """Persist a concise, objective memory of the agent execution."""
        try:
            record = dispatch.record
            content = (
                f"Agent '{dispatch.definition.name}' (v{dispatch.definition.version}) "
                f"handled task '{record.task_id}' with status '{record.final_status}'. "
                f"Success={record.success}, iterations={record.iterations_used}, "
                f"tool_calls={record.tool_calls}."
            )
            from app.memory.models import MemorySource, MemoryType

            stored = await self._memory.add(
                content=content,
                memory_type=MemoryType.EPISODIC,
                task_id=record.task_id,
                session_id=record.session_id,
                agent_id=record.agent_id,
                source=MemorySource.SYSTEM,
                importance=0.6 if record.success else 0.4,
                confidence=0.8 if record.success else 0.5,
                summary=f"Agent {record.agent_id} {record.final_status}",
                metadata={
                    "agent_id": record.agent_id,
                    "agent_version": record.agent_version,
                    "success": record.success,
                    "iterations": record.iterations_used,
                },
            )
            return stored.memory_id if stored else None
        except Exception as exc:  # noqa: BLE001 - memory persistence is best-effort
            logger.bind(event="agent.orchestration.memory.error", error=type(exc).__name__).debug(
                "Memory persistence failed: {}", str(exc)
            )
            return None
