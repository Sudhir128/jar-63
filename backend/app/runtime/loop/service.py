"""Loop service: ties tasks to the Universal Loop Engine.

Bridges the :class:`TaskManager` and the :class:`LoopController`. Creates a
task, builds a loop context from runtime collaborators, runs the loop
(optionally in the background), and tracks active loops for status, events,
and cancellation.

Phase 3: the service optionally wires the LLM planner (when ``LLM_ENABLED``)
and the :class:`ToolExecutor` (with policy) into the loop controller. The
deterministic planner remains the fallback.

Phase 6: when a :class:`MemoryManager` is wired, the discover stage is
wrapped in :class:`MemoryDiscoverStage` so a bounded memory context is
retrieved for each task goal, and completed loop results are persisted as
memories (task/episodic) for future retrieval.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from app.agents.registry import AgentRegistry
from app.config import Settings, get_settings
from app.core.identifiers import utc_now
from app.core.logging import get_logger
from app.events import Event, EventBus, EventType
from app.runtime.loop.loop_context import LoopContext
from app.runtime.loop.loop_controller import LoopController
from app.runtime.loop.loop_policy import LoopPolicy
from app.runtime.loop.loop_result import LoopFinalStatus, LoopResult
from app.runtime.loop.loop_state import LoopState
from app.runtime.loop.stages import (
    DefaultDiscoverStage,
    DefaultExecuteStage,
    DefaultIterateStage,
    DefaultPlanStage,
    DefaultVerifyStage,
    MemoryDiscoverStage,
)
from app.runtime.loop.stages.plan import LLMPlanStage
from app.runtime.loop.verification import Verifier
from app.runtime.models import Task, TaskStatus
from app.runtime.session_manager import SessionManager
from app.runtime.task_manager import TaskManager
from app.tools.confirmation import ConfirmationRequest, ConfirmationStore
from app.tools.executor import ToolExecutor
from app.tools.policy import DefaultToolPolicy, ToolPolicy
from app.tools.registry import ToolRegistry

logger = get_logger("loop.service")

__all__ = ["LoopService", "LoopRunHandle"]


@dataclass
class LoopRunHandle:
    """Handle to a running or completed loop, keyed by task_id."""

    loop_id: str
    task_id: str
    controller: LoopController
    context: LoopContext
    background_task: asyncio.Task | None = None
    result: LoopResult | None = None
    started_at: Any = None
    confirmation_store: ConfirmationStore | None = None


class LoopService:
    """Manages loop lifecycles for tasks.

    Loops run in the background via :meth:`start_task_loop`; their state is
    queryable and cancellable through this service.
    """

    def __init__(
        self,
        *,
        task_manager: TaskManager,
        agent_registry: AgentRegistry,
        tool_registry: ToolRegistry,
        event_bus: EventBus,
        session_manager: SessionManager | None = None,
        settings: Settings | None = None,
        tool_policy: ToolPolicy | None = None,
        llm_planner: Any | None = None,
        memory_manager: Any | None = None,
    ) -> None:
        self.task_manager = task_manager
        self.agent_registry = agent_registry
        self.tool_registry = tool_registry
        self.event_bus = event_bus
        self.session_manager = session_manager
        self.settings = settings or get_settings()
        self.tool_policy = tool_policy or DefaultToolPolicy()
        self.llm_planner = llm_planner
        self.memory_manager = memory_manager
        self._confirmation_store = ConfirmationStore()
        self._tool_executor = ToolExecutor(
            registry=self.tool_registry,
            policy=self.tool_policy,
            confirmation_store=self._confirmation_store,
            event_bus=self.event_bus,
        )
        self._handles: dict[str, LoopRunHandle] = {}

    @property
    def confirmation_store(self) -> ConfirmationStore:
        return self._confirmation_store

    def make_controller(
        self,
        *,
        verifier: Verifier | None = None,
        stop_conditions: list | None = None,
    ) -> LoopController:
        """Build a :class:`LoopController` with Phase 3 stages.

        When an LLM planner is configured and ``LLM_ENABLED``, the LLM plan
        stage is used (with deterministic fallback). Otherwise the default
        deterministic planner is used. The tool executor is always wired so
        tool calls flow through policy.
        """
        plan_stage: Any = DefaultPlanStage()
        if self.llm_planner is not None and self.settings.llm.enabled:
            plan_stage = LLMPlanStage(self.llm_planner)

        # Phase 6: wrap discover with memory retrieval when a manager is wired.
        discover_stage: Any = DefaultDiscoverStage()
        if self.memory_manager is not None:
            discover_stage = MemoryDiscoverStage(self.memory_manager)

        return LoopController(
            discover_stage=discover_stage,
            plan_stage=plan_stage,
            execute_stage=DefaultExecuteStage(tool_executor=self._tool_executor),
            verify_stage=DefaultVerifyStage(verifier=verifier),
            iterate_stage=DefaultIterateStage(),
            stop_conditions=stop_conditions,
        )

    def _build_context(
        self,
        task: Task,
        *,
        goal: str,
        success_criteria: list[str],
        max_iterations: int,
        expected_output: Any,
        verifier: Verifier | None,
    ) -> tuple[LoopController, LoopContext]:
        controller = self.make_controller(verifier=verifier)
        policy = LoopPolicy(max_iterations=max_iterations)
        state = LoopState(
            task_id=task.task_id,
            session_id=task.session_id,
            goal=goal,
            success_criteria=list(success_criteria),
            max_iterations=max_iterations,
        )
        context = LoopContext(
            state=state,
            task=task,
            agent_registry=self.agent_registry,
            tool_registry=self.tool_registry,
            event_bus=self.event_bus,
            policy=policy,
            settings=self.settings,
            stage_config={"expected_output": expected_output},
        )
        return controller, context

    async def run_task_loop(
        self,
        task: Task,
        *,
        goal: str | None = None,
        success_criteria: list[str] | None = None,
        max_iterations: int = 5,
        expected_output: Any = None,
        verifier: Verifier | None = None,
    ) -> LoopResult:
        """Run a loop synchronously (inline) and return the result."""
        self.task_manager.register(task)
        controller, context = self._build_context(
            task,
            goal=goal or str(task.input),
            success_criteria=success_criteria or [],
            max_iterations=max_iterations,
            expected_output=expected_output,
            verifier=verifier,
        )
        handle = LoopRunHandle(
            loop_id=context.loop_id,
            task_id=task.task_id,
            controller=controller,
            context=context,
            started_at=utc_now(),
            confirmation_store=self._confirmation_store,
        )
        self._handles[task.task_id] = handle
        task.status = TaskStatus.RUNNING
        task.updated_at = utc_now()
        result = await controller.run(context)
        handle.result = result
        await self._update_task_status(task.task_id, result)
        return result

    async def start_task_loop(
        self,
        task: Task,
        *,
        goal: str | None = None,
        success_criteria: list[str] | None = None,
        max_iterations: int = 5,
        expected_output: Any = None,
        verifier: Verifier | None = None,
    ) -> LoopRunHandle:
        """Start a loop in the background and return a handle immediately."""
        self.task_manager.register(task)
        controller, context = self._build_context(
            task,
            goal=goal or str(task.input),
            success_criteria=success_criteria or [],
            max_iterations=max_iterations,
            expected_output=expected_output,
            verifier=verifier,
        )
        handle = LoopRunHandle(
            loop_id=context.loop_id,
            task_id=task.task_id,
            controller=controller,
            context=context,
            started_at=utc_now(),
            confirmation_store=self._confirmation_store,
        )
        self._handles[task.task_id] = handle
        task.status = TaskStatus.RUNNING
        task.updated_at = utc_now()

        async def _runner() -> None:
            try:
                result = await controller.run(context)
                handle.result = result
                await self._update_task_status(task.task_id, result)
            except Exception as exc:  # noqa: BLE001
                logger.bind(event="loop.bg.error", task_id=task.task_id).exception(
                    "Background loop failed: {}", str(exc)
                )
                handle.result = LoopResult(
                    loop_id=context.loop_id,
                    task_id=task.task_id,
                    session_id=task.session_id,
                    final_status=LoopFinalStatus.FAILED,
                    failure_reason=str(exc),
                )
                await self._update_task_status(task.task_id, handle.result)

        handle.background_task = asyncio.create_task(_runner())
        return handle

    def get_handle(self, task_id: str) -> LoopRunHandle | None:
        return self._handles.get(task_id)

    def get_state(self, task_id: str) -> LoopState | None:
        handle = self._handles.get(task_id)
        return handle.controller.state if handle else None

    def cancel(self, task_id: str) -> bool:
        handle = self._handles.get(task_id)
        if handle is None:
            return False
        handle.controller.request_cancel()
        return True

    def list_pending_confirmations(self) -> list[dict[str, Any]]:
        """Return all pending tool confirmation requests."""
        return [r.model_dump() for r in self._confirmation_store.list_pending()]

    async def approve_confirmation(
        self, confirmation_id: str, *, decided_by: str | None = None
    ) -> bool:
        """Approve a pending tool confirmation and publish the event."""
        req = self._confirmation_store.approve(confirmation_id, decided_by=decided_by)
        if req is None:
            return False
        await self._publish_confirmation_event(EventType.TOOL_CONFIRMATION_APPROVED, req)
        return True

    async def reject_confirmation(
        self, confirmation_id: str, *, decided_by: str | None = None
    ) -> bool:
        """Reject a pending tool confirmation and publish the event."""
        req = self._confirmation_store.reject(confirmation_id, decided_by=decided_by)
        if req is None:
            return False
        await self._publish_confirmation_event(EventType.TOOL_CONFIRMATION_REJECTED, req)
        return True

    async def resume_loop(self, task_id: str) -> LoopResult:
        """Resume a paused loop after its confirmation was approved.

        The same task_id, loop_id, session_id, and iteration continue. No
        second task is created. The confirmed tool is executed, then the loop
        proceeds from VERIFY → DECIDE → (ITERATE or done).
        """
        handle = self._handles.get(task_id)
        if handle is None:
            raise ValueError(f"No loop found for task '{task_id}'.")
        state = handle.context.state
        if not state.confirmation_required or not state.confirmation_request:
            raise ValueError(f"Task '{task_id}' is not waiting for confirmation.")
        confirmation_id = state.confirmation_request.get("confirmation_id")
        if not confirmation_id:
            raise ValueError("No confirmation_id in the pending request.")
        req = self._confirmation_store.get(confirmation_id)
        if req is None or not req.is_approved:
            raise ValueError(
                f"Confirmation '{confirmation_id}' is not approved (status="
                f"{req.status.value if req else 'missing'})."
            )
        result = await handle.controller.resume_after_confirmation(handle.context, confirmation_id)
        handle.result = result
        await self._update_task_status(task_id, result)
        return result

    async def resume_loop_after_rejection(self, task_id: str, *, reason: str = "") -> LoopResult:
        """Resume a paused loop after its confirmation was rejected.

        The rejected tool is NOT executed. The loop proceeds from VERIFY
        (which will fail) → DECIDE → ITERATE, or fails safely.
        """
        handle = self._handles.get(task_id)
        if handle is None:
            raise ValueError(f"No loop found for task '{task_id}'.")
        state = handle.context.state
        if not state.confirmation_required or not state.confirmation_request:
            raise ValueError(f"Task '{task_id}' is not waiting for confirmation.")
        confirmation_id = state.confirmation_request.get("confirmation_id")
        if not confirmation_id:
            raise ValueError("No confirmation_id in the pending request.")
        result = await handle.controller.resume_after_rejection(
            handle.context, confirmation_id, reason=reason
        )
        handle.result = result
        await self._update_task_status(task_id, result)
        return result

    async def _publish_confirmation_event(
        self, event_type: EventType, req: ConfirmationRequest
    ) -> None:
        await self.event_bus.publish(
            Event.create(
                event_type,
                task_id=req.task_id,
                session_id=req.session_id,
                payload={
                    "confirmation_id": req.confirmation_id,
                    "tool": req.tool_name,
                    "risk": req.risk_level.value,
                    "reason": req.reason,
                    "decided_by": req.decided_by,
                    "loop_id": req.loop_id,
                },
                metadata={
                    "confirmation_id": req.confirmation_id,
                    "tool": req.tool_name,
                },
            )
        )

    def list_loops(self) -> list[dict[str, Any]]:
        return [
            {
                "loop_id": h.loop_id,
                "task_id": h.task_id,
                "status": h.controller.state.status.value,
                "iteration_count": h.controller.state.iteration_count,
                "final_status": h.result.final_status.value if h.result else None,
            }
            for h in self._handles.values()
        ]

    async def _update_task_status(self, task_id: str, result: LoopResult) -> None:
        status_map = {
            LoopFinalStatus.SUCCESS: TaskStatus.COMPLETED,
            LoopFinalStatus.CANCELLED: TaskStatus.CANCELLED,
            LoopFinalStatus.FAILED: TaskStatus.FAILED,
            LoopFinalStatus.MAX_ITERATIONS_REACHED: TaskStatus.FAILED,
            LoopFinalStatus.INCOMPLETE: TaskStatus.FAILED,
            LoopFinalStatus.WAITING_FOR_CONFIRMATION: TaskStatus.WAITING_FOR_CONFIRMATION,
        }
        task_status = status_map.get(result.final_status, TaskStatus.FAILED)
        task = self.task_manager.find(task_id)
        if task is not None:
            # Mutate the tracked task in place so callers observe the update.
            task.status = task_status
            task.result = result.final_response
            task.error = result.failure_reason
            task.updated_at = utc_now()
        # Phase 6: persist a memory of completed/failed loops (non-fatal).
        await self._persist_loop_memory(task, result)

    async def _persist_loop_memory(self, task: Task | None, result: LoopResult) -> None:
        """Persist a TASK memory summarizing the loop outcome.

        Only terminal, non-confirmation statuses are persisted. Memory writes
        are best-effort and never fail the loop.
        """
        if self.memory_manager is None or not self.memory_manager.is_enabled:
            return
        if result.final_status is LoopFinalStatus.WAITING_FOR_CONFIRMATION:
            return
        if task is None:
            return
        from app.memory.types import MemorySource, MemoryType

        goal = task.input if isinstance(task.input, str) else str(task.input)
        status_label = result.final_status.value
        content = f"Task '{goal}' ended with status '{status_label}'."
        if result.failure_reason:
            content += f" Reason: {result.failure_reason}."
        if result.final_response is not None:
            resp = str(result.final_response)[:200]
            content += f" Result: {resp}."
        importance = 0.7 if result.success else 0.4
        try:
            await self.memory_manager.add(
                content=content,
                memory_type=MemoryType.TASK,
                session_id=result.session_id,
                task_id=result.task_id,
                source=MemorySource.SYSTEM,
                importance=importance,
                confidence=0.8,
                summary=f"Loop {result.loop_id}: {status_label}",
                metadata={
                    "loop_id": result.loop_id,
                    "iterations": result.iterations_used,
                    "final_status": status_label,
                },
            )
        except Exception as exc:  # noqa: BLE001 - memory must never fail the loop
            logger.bind(
                event="loop.memory.persist.error",
                error=type(exc).__name__,
                task_id=result.task_id,
            ).warning("Failed to persist loop memory (non-fatal): {}", str(exc))
