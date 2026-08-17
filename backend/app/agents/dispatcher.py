"""Agent dispatcher (Phase 7).

Resolves a definition, constructs an ephemeral runtime instance, injects
memory context, executes through the *existing* :class:`LoopService` (the Loop
Engine remains the execution backbone), collects the loop result into an
objective :class:`AgentExecutionRecord`, persists the evidence, and updates
usage metadata. It never duplicates the loop engine.

Flow::

    RoutingDecision (agent_id)
        ↓
    resolve AgentDefinition from catalog
        ↓
    construct runtime instance (AgentFactory)
        ↓
    register instance into the existing AgentRegistry (runtime lookup)
        ↓
    memory context (Phase 6, bounded, relevant)
        ↓
    set task.agent_id
        ↓
    LoopService.run_task_loop (existing Loop Engine)
        ↓
    LoopResult → AgentExecutionRecord (objective evidence)
        ↓
    persist + update usage + release runtime instance
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.agents.definitions import AgentDefinition, AgentExecutionRecord
from app.agents.factory import AgentFactory
from app.core.exceptions import AgentNotDispatchableError
from app.core.identifiers import utc_now
from app.core.logging import get_logger
from app.events import Event, EventBus, EventType
from app.runtime.models import Task

if TYPE_CHECKING:
    from app.agents.catalog import AgentDefinitionRegistry
    from app.agents.store import AgentExecutionStore
    from app.memory.manager import MemoryManager
    from app.runtime.loop.service import LoopService

logger = get_logger("agents.dispatcher")

__all__ = ["AgentDispatcher", "DispatchResult"]


@dataclass
class DispatchResult:
    """The outcome of an agent dispatch."""

    definition: AgentDefinition
    task: Task
    loop_result: Any | None
    record: AgentExecutionRecord
    memory_items_used: int = 0


class AgentDispatcher:
    """Executes an agent definition through the existing Loop Engine."""

    def __init__(
        self,
        catalog: AgentDefinitionRegistry,
        execution_store: AgentExecutionStore,
        *,
        agent_registry: Any,
        loop_service: LoopService,
        factory: AgentFactory | None = None,
        memory_manager: MemoryManager | None = None,
        event_bus: EventBus | None = None,
        ephemeral: bool = True,
    ) -> None:
        self._catalog = catalog
        self._exec_store = execution_store
        self._agent_registry = agent_registry
        self._loop_service = loop_service
        self._factory = factory or AgentFactory(
            tool_executor=loop_service._tool_executor  # noqa: SLF001 - shared executor
        )
        self._memory = memory_manager
        self._event_bus = event_bus
        # When True, the runtime instance is unregistered after the dispatch
        # (ephemeral). The definition persists in the catalog regardless.
        self._ephemeral = ephemeral

    async def dispatch(
        self,
        agent_id: str,
        task: Task,
        *,
        goal: str | None = None,
        success_criteria: list[str] | None = None,
        max_iterations: int = 5,
        expected_output: Any = None,
        verifier: Any | None = None,
        session_id: str | None = None,
    ) -> DispatchResult:
        definition = await self._catalog.get(agent_id)
        if not definition.is_dispatchable:
            raise AgentNotDispatchableError(
                f"Agent '{agent_id}' is not dispatchable (lifecycle={definition.lifecycle.value})"
            )
        await self._publish_started(definition, task)

        instance = self._factory.build(definition)
        # Register the runtime instance into the existing AgentRegistry so the
        # loop's execute stage can resolve it. If one is already present (shared
        # built-in agent), keep it.
        from app.core.exceptions import AgentAlreadyRegisteredError

        already_present = self._agent_registry.exists(instance.agent_id)
        if not already_present:
            try:
                await self._agent_registry.register(instance)
            except AgentAlreadyRegisteredError:
                already_present = True

        # Inject relevant memory context (bounded, privacy-respecting).
        memory_items = await self._inject_memory(definition, task, goal or str(task.input))

        task.agent_id = instance.agent_id
        if session_id and not task.session_id:
            task.session_id = session_id

        started = time.perf_counter()
        loop_result = None
        try:
            # When the caller provides an expected output but no verifier,
            # use an exact-match verifier so the loop can objectively verify
            # the agent's result. This keeps dispatch self-contained: callers
            # don't have to know about the loop's verifier protocol.
            effective_verifier = verifier
            if effective_verifier is None and expected_output is not None:
                from app.runtime.loop.verification.verifier import ExactMatchVerifier

                effective_verifier = ExactMatchVerifier(check_name="agent_exact_match")
            loop_result = await self._loop_service.run_task_loop(
                task,
                goal=goal or str(task.input),
                success_criteria=success_criteria or [],
                max_iterations=max_iterations,
                expected_output=expected_output,
                verifier=effective_verifier,
            )
        except Exception as exc:  # noqa: BLE001 - never fabricate; record failure
            logger.bind(
                event="agent.dispatch.exception", agent_id=agent_id, error=type(exc).__name__
            ).warning("Dispatch raised: {}", str(exc))
            loop_result = None
            failure = str(exc)
        else:
            failure = None
        latency_ms = int((time.perf_counter() - started) * 1000)

        record = self._build_record(
            definition, task, loop_result, latency_ms, memory_items, failure=failure
        )
        await self._exec_store.add(record)
        await self._update_usage(definition, record)
        await self._publish_completed(definition, task, record)

        # Release the ephemeral runtime instance (definition persists).
        if self._ephemeral and not already_present and definition.dynamic:
            with contextlib.suppress(Exception):  # noqa: BLE001 - cleanup must never fail dispatch
                await self._agent_registry.unregister(instance.agent_id)

        return DispatchResult(
            definition=definition,
            task=task,
            loop_result=loop_result,
            record=record,
            memory_items_used=memory_items,
        )

    # --- memory ---------------------------------------------------------

    async def _inject_memory(self, definition: AgentDefinition, task: Task, goal: str) -> int:
        """Retrieve a bounded, relevant memory context onto the task metadata.

        Only relevant memory is injected (never the whole DB). Respects privacy
        (scoped to the task/session/agent) and context limits (the retriever
        packs a bounded context).
        """
        if self._memory is None or not self._memory.is_enabled:
            return 0
        try:
            ctx = await self._memory.retrieve_context(
                goal,
                session_id=task.session_id,
                task_id=task.task_id,
                agent_id=definition.agent_id,
            )
            task.metadata["memory_context"] = ctx.model_dump(mode="json")
            return len(ctx.memories)
        except Exception as exc:  # noqa: BLE001 - memory must never break dispatch
            logger.bind(event="agent.dispatch.memory.error", error=type(exc).__name__).debug(
                "Memory injection failed (non-fatal): {}", str(exc)
            )
            return 0

    # --- evidence -------------------------------------------------------

    def _build_record(
        self,
        definition: AgentDefinition,
        task: Task,
        loop_result: Any | None,
        latency_ms: int,
        memory_items: int,
        *,
        failure: str | None = None,
    ) -> AgentExecutionRecord:
        success = bool(loop_result and loop_result.success)
        final_status = loop_result.final_status.value if loop_result else "failed"
        iterations = loop_result.iterations_used if loop_result else 0
        # Tool stats from the loop state (if available).
        tool_calls = 0
        tool_failures = 0
        policy_violations = 0
        confirmations = 0
        retries = max(0, iterations - 1) if iterations else 0
        cancelled = bool(loop_result and loop_result.final_status.value == "cancelled")
        failure_reason = failure
        if loop_result and loop_result.failure_reason:
            failure_reason = loop_result.failure_reason
        handle = self._loop_service.get_handle(task.task_id)
        if handle is not None:
            state = handle.controller.state
            tool_calls = state.tool_call_count
            tool_failures = sum(1 for r in state.execution_results if r.status != "success")
            policy_violations = sum(
                1
                for o in state.observations
                if isinstance(o, dict) and o.get("type") == "tool_denied"
            )
            confirmations = 1 if state.confirmation_required else 0
        return AgentExecutionRecord(
            agent_id=definition.agent_id,
            agent_version=definition.version,
            task_id=task.task_id,
            session_id=task.session_id,
            loop_id=handle.loop_id if handle else None,
            success=success,
            final_status=final_status,
            iterations_used=iterations,
            tool_calls=tool_calls,
            tool_failures=tool_failures,
            policy_violations=policy_violations,
            confirmations_requested=confirmations,
            retries=retries,
            cancelled=cancelled,
            latency_ms=latency_ms,
            failure_reason=failure_reason,
            memory_items_used=memory_items,
            metadata={"dynamic": definition.dynamic, "kind": definition.kind.value},
        )

    async def _update_usage(
        self, definition: AgentDefinition, record: AgentExecutionRecord
    ) -> None:
        u = definition.usage
        new_dispatch = u.dispatch_count + 1
        new_success = u.success_count + (1 if record.success else 0)
        new_failure = u.failure_count + (0 if record.success else 1)
        now = utc_now()
        from app.agents.definitions import AgentUsageMetadata

        updated = AgentUsageMetadata(
            dispatch_count=new_dispatch,
            success_count=new_success,
            failure_count=new_failure,
            last_dispatched_at=now,
            last_success_at=now if record.success else u.last_success_at,
            last_failure_at=now if not record.success else u.last_failure_at,
        )
        await self._catalog.update_usage(definition.agent_id, updated)

    # --- events ---------------------------------------------------------

    async def _publish_started(self, definition: AgentDefinition, task: Task) -> None:
        if self._event_bus is None:
            return
        await self._event_bus.publish(
            Event.create(
                EventType.AGENT_DISPATCH_STARTED,
                agent_id=definition.agent_id,
                task_id=task.task_id,
                session_id=task.session_id,
                payload={
                    "version": definition.version,
                    "kind": definition.kind.value,
                    "dynamic": definition.dynamic,
                },
                metadata={"agent_id": definition.agent_id, "version": definition.version},
            )
        )

    async def _publish_completed(
        self, definition: AgentDefinition, task: Task, record: AgentExecutionRecord
    ) -> None:
        if self._event_bus is None:
            return
        event_type = (
            EventType.AGENT_DISPATCH_COMPLETED
            if record.success
            else EventType.AGENT_DISPATCH_FAILED
        )
        await self._event_bus.publish(
            Event.create(
                event_type,
                agent_id=definition.agent_id,
                task_id=task.task_id,
                session_id=task.session_id,
                payload={
                    "success": record.success,
                    "final_status": record.final_status,
                    "iterations": record.iterations_used,
                    "tool_calls": record.tool_calls,
                    "tool_failures": record.tool_failures,
                    "policy_violations": record.policy_violations,
                    "latency_ms": record.latency_ms,
                    "version": record.agent_version,
                },
                metadata={
                    "agent_id": definition.agent_id,
                    "version": record.agent_version,
                    "loop_id": record.loop_id,
                },
            )
        )
