"""Execute stage.

Performs the current planned action using the :class:`AgentRegistry` and
:class:`ToolRegistry` — never instantiating agents/tools directly. Supports
async execution, per-execution timeout, errors, and cancellation, and
publishes events throughout.

Phase 3 extends tool execution to flow through the :class:`ToolExecutor`:
the LLM/plan may *request* a tool call, but actual execution goes through
policy validation, registry resolution, schema validation, and the executor.
The LLM never executes tools directly.

Maker / Checker note: the execute stage (the maker) only *produces* output.
Whether that output satisfies the objective is decided independently by the
Verify stage (the checker).
"""

from __future__ import annotations

import time
from typing import Any

import anyio

from app.agents.interface import AgentContext, AgentResult, AgentStatus
from app.core.identifiers import utc_now
from app.core.logging import get_logger
from app.events import Event, EventType
from app.runtime.loop.loop_context import LoopContext
from app.runtime.loop.loop_errors import LoopCancelledError
from app.runtime.loop.loop_state import (
    ActionType,
    ExecutionResult,
    LoopStatus,
    StageStatus,
)
from app.runtime.loop.stages.base import LoopStage
from app.tools.executor import ToolCallRecord, ToolExecutionOutcome, ToolExecutor
from app.tools.interface import ToolContext, ToolResult
from app.tools.policy import AllowAllToolPolicy, ToolPolicy

logger = get_logger("loop.execute")

__all__ = ["ExecuteStage", "DefaultExecuteStage"]


class ExecuteStage(LoopStage):
    """Abstract execute stage contract."""

    name = "execute"

    async def run(self, context: LoopContext) -> LoopContext:
        result = await self.execute(context)
        context.update_state(
            current_stage=StageStatus.EXECUTE,
            status=LoopStatus.EXECUTING,
            last_execution=result,
            execution_results=[*context.state.execution_results, result],
        )
        if result.status == "success":
            steps = [*context.state.completed_steps, result.execution_id]
            context.update_state(completed_steps=steps)
        else:
            failed = [*context.state.failed_steps, result.execution_id]
            context.update_state(failed_steps=failed, last_error=result.error)
        return context

    async def execute(self, context: LoopContext) -> ExecutionResult:
        """Perform the planned action and return an execution result."""


class DefaultExecuteStage(ExecuteStage):
    """Default execute stage backed by the agent and tool registries.

    Resolves the next action's target through the registries. It will never
    execute an agent or tool that is not registered.

    Phase 3: when a :class:`ToolExecutor` is available, tool execution flows
    through it (policy → registry → schema → execute). When no executor is
    configured, tools are executed directly (backward compatibility with
    Phase 0/1 behavior).
    """

    def __init__(
        self,
        *,
        tool_executor: ToolExecutor | None = None,
        tool_policy: ToolPolicy | None = None,
    ) -> None:
        self._tool_executor = tool_executor
        self._tool_policy = tool_policy or AllowAllToolPolicy()

    async def execute(self, context: LoopContext) -> ExecutionResult:
        # Honour cancellation before starting any work.
        if context.state.cancel_requested:
            return _cancelled_result(context, "cancelled before execution")

        action = context.state.next_action
        if action is None or action.action_type is ActionType.NONE:
            return ExecutionResult(
                status="skipped",
                error="No action to execute.",
            )

        if action.action_type is ActionType.EXECUTE_AGENT:
            return await self._execute_agent(context, action.target, action.parameters)
        if action.action_type is ActionType.EXECUTE_TOOL:
            return await self._execute_tool(context, action.target, action.parameters)
        return ExecutionResult(
            status="skipped",
            error=f"Unsupported action type: {action.action_type}",
        )

    async def _execute_agent(
        self, context: LoopContext, agent_id: str | None, params: dict[str, Any]
    ) -> ExecutionResult:
        if not agent_id:
            return ExecutionResult(status="failed", error="No agent id in plan.")
        if not context.agent_registry.exists(agent_id):
            return ExecutionResult(
                agent_id=agent_id,
                status="failed",
                error=f"Unknown agent: {agent_id}",
            )
        agent = context.agent_registry.get(agent_id)
        agent_context = AgentContext(
            task_id=context.task_id,
            session_id=context.session_id,
            input=context.task.input,
            metadata={**params, "iteration": context.state.iteration_count},
        )
        await context.event_bus.publish(
            _event(
                context,
                EventType.AGENT_STARTED,
                agent_id=agent_id,
                payload={"iteration": context.state.iteration_count},
            )
        )
        started = time.monotonic()
        try:
            with anyio.fail_after(context.policy.per_execution_timeout_seconds):
                await agent.on_start(agent_context)
                result: AgentResult = await agent.execute(agent_context)
                await agent.on_complete(result)
        except TimeoutError as exc:
            await agent.on_error(exc)
            await context.event_bus.publish(
                _event(
                    context,
                    EventType.AGENT_FAILED,
                    agent_id=agent_id,
                    payload={"error": "timeout"},
                )
            )
            return ExecutionResult(
                agent_id=agent_id,
                status="timeout",
                error=f"Agent timed out: {exc}",
                completed_at=utc_now(),
                duration_ms=int((time.monotonic() - started) * 1000),
            )
        except LoopCancelledError as exc:
            await agent.on_error(exc)
            return _cancelled_result(context, str(exc), agent_id=agent_id)
        except Exception as exc:  # noqa: BLE001 - isolate execution failures
            await agent.on_error(exc)
            await context.event_bus.publish(
                _event(
                    context,
                    EventType.AGENT_FAILED,
                    agent_id=agent_id,
                    payload={"error": str(exc)},
                )
            )
            return ExecutionResult(
                agent_id=agent_id,
                status="failed",
                error=str(exc),
                completed_at=utc_now(),
                duration_ms=int((time.monotonic() - started) * 1000),
            )

        status = "success" if result.status is AgentStatus.COMPLETED else "failed"
        # When an agent signals WAITING_FOR_CONFIRMATION (e.g. its tool call
        # requires confirmation), pause the loop instead of treating it as a
        # failure. The confirmation request is carried in result.metadata.
        if result.status is AgentStatus.WAITING_FOR_CONFIRMATION:
            status = "confirmation_required"
            confirm_data = result.metadata.get("confirmation_request")
            if confirm_data:
                context.update_state(
                    confirmation_required=True,
                    confirmation_request={**confirm_data, "loop_id": context.loop_id},
                )
        await context.event_bus.publish(
            _event(
                context,
                EventType.AGENT_COMPLETED,
                agent_id=agent_id,
                payload={"status": result.status.value},
            )
        )
        return ExecutionResult(
            agent_id=agent_id,
            status=status,
            output=result.output,
            error=result.error,
            completed_at=utc_now(),
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    async def _execute_tool(
        self, context: LoopContext, tool_name: str | None, params: dict[str, Any]
    ) -> ExecutionResult:
        if not tool_name:
            return ExecutionResult(status="failed", error="No tool name in plan.")

        # If a ToolExecutor is configured, route through the full controlled
        # pipeline (policy → registry → schema → execute → observation).
        if self._tool_executor is not None:
            return await self._execute_tool_via_executor(context, tool_name, params)

        # Backward-compatible path: execute directly through the registry.
        return await self._execute_tool_direct(context, tool_name, params)

    async def _execute_tool_via_executor(
        self, context: LoopContext, tool_name: str, params: dict[str, Any]
    ) -> ExecutionResult:
        call = ToolCallRecord(
            tool_call_id=str(params.get("tool_call_id") or f"iter_{context.state.iteration_count}"),
            tool_name=tool_name,
            arguments=dict(params.get("arguments", params)),
        )
        # Check tool-call limits.
        state = context.state
        if state.tool_call_count >= context.policy.max_tool_calls_per_task:
            return ExecutionResult(
                tool_name=tool_name,
                status="failed",
                error=f"Tool call limit reached (max {context.policy.max_tool_calls_per_task} per task).",
            )
        if state.tool_call_count_per_iteration >= context.policy.max_tool_calls_per_iteration:
            return ExecutionResult(
                tool_name=tool_name,
                status="failed",
                error=(
                    f"Tool call limit reached (max "
                    f"{context.policy.max_tool_calls_per_iteration} per iteration)."
                ),
            )

        # Repeated-call protection.
        self._tool_executor.record_call(context.task_id, call)
        repeats = self._tool_executor.repeated_call_count(context.task_id, call)
        if repeats > context.policy.max_repeated_identical_tool_calls:
            return ExecutionResult(
                tool_name=tool_name,
                status="failed",
                error=(
                    f"Repeated identical tool call '{tool_name}' exceeded limit "
                    f"({context.policy.max_repeated_identical_tool_calls})."
                ),
            )

        outcome: ToolExecutionOutcome = await self._tool_executor.execute_call(
            call,
            task_id=context.task_id,
            session_id=context.session_id,
            iteration=context.state.iteration_count,
            timeout_seconds=context.policy.per_execution_timeout_seconds,
        )

        # Record the observation into loop state.
        obs = outcome.observation
        context.update_state(
            observations=[*state.observations, obs.model_dump()],
            tool_call_count=state.tool_call_count + 1,
            tool_call_count_per_iteration=state.tool_call_count_per_iteration + 1,
        )

        # Handle confirmation-required: pause execution.
        if outcome.confirmation is not None:
            context.update_state(
                confirmation_required=True,
                confirmation_request={
                    **outcome.confirmation.model_dump(),
                    "loop_id": context.loop_id,
                },
            )
            return ExecutionResult(
                tool_name=tool_name,
                status="confirmation_required",
                error=f"Tool '{tool_name}' requires confirmation: {outcome.confirmation.reason}",
                completed_at=utc_now(),
            )

        if outcome.skipped:
            return ExecutionResult(
                tool_name=tool_name,
                status="failed",
                error=outcome.skipped_reason or f"Tool '{tool_name}' was skipped.",
                completed_at=utc_now(),
            )

        result = outcome.result
        return ExecutionResult(
            tool_name=tool_name,
            status="success" if result.success else "failed",
            output=result.output,
            error=result.error,
            completed_at=utc_now(),
            duration_ms=result.execution_time_ms,
        )

    async def execute_confirmed(
        self, context: LoopContext, confirmation_id: str
    ) -> ExecutionResult:
        """Execute a previously-confirmed tool call during loop resume.

        Called by :meth:`LoopController.resume_after_confirmation`. The tool
        is executed directly (policy already satisfied at confirmation time),
        and the result is normalized into an :class:`ExecutionResult`.
        """
        outcome: ToolExecutionOutcome = await self._tool_executor.execute_confirmed(
            confirmation_id,
            task_id=context.task_id,
            session_id=context.session_id,
            iteration=context.state.iteration_count,
            timeout_seconds=context.policy.per_execution_timeout_seconds,
        )
        obs = outcome.observation
        state = context.state
        context.update_state(
            observations=[*state.observations, obs.model_dump()],
            tool_call_count=state.tool_call_count + 1,
            tool_call_count_per_iteration=state.tool_call_count_per_iteration + 1,
            confirmation_required=False,
            confirmation_request=None,
        )
        if outcome.skipped or outcome.result is None:
            return ExecutionResult(
                tool_name=obs.tool_name or "unknown",
                status="failed",
                error=outcome.skipped_reason or "Confirmed tool could not execute.",
                completed_at=utc_now(),
            )
        result = outcome.result
        return ExecutionResult(
            tool_name=result.name,
            status="success" if result.success else "failed",
            output=result.output,
            error=result.error,
            completed_at=utc_now(),
            duration_ms=result.execution_time_ms,
        )

    @staticmethod
    def rejected_result(context: LoopContext, reason: str) -> ExecutionResult:
        """Build an execution result for a rejected confirmation."""
        confirm = context.state.confirmation_request or {}
        tool_name = confirm.get("tool_name", "unknown")
        return ExecutionResult(
            tool_name=tool_name,
            status="rejected",
            error=f"Tool '{tool_name}' was rejected: {reason}",
            completed_at=utc_now(),
        )

    async def _execute_tool_direct(
        self, context: LoopContext, tool_name: str, params: dict[str, Any]
    ) -> ExecutionResult:
        if not context.tool_registry.exists(tool_name):
            return ExecutionResult(
                tool_name=tool_name,
                status="failed",
                error=f"Unknown tool: {tool_name}",
            )
        tool = context.tool_registry.get(tool_name)
        tool_context = ToolContext(
            task_id=context.task_id,
            session_id=context.session_id,
            arguments=params,
            metadata={"iteration": context.state.iteration_count},
        )
        await context.event_bus.publish(
            _event(
                context,
                EventType.TOOL_STARTED,
                payload={"tool": tool_name},
            )
        )
        started = time.monotonic()
        try:
            with anyio.fail_after(context.policy.per_execution_timeout_seconds):
                result: ToolResult = await tool.execute(tool_context)
        except TimeoutError as exc:
            await context.event_bus.publish(
                _event(
                    context,
                    EventType.TOOL_COMPLETED,
                    payload={"tool": tool_name, "error": "timeout"},
                )
            )
            return ExecutionResult(
                tool_name=tool_name,
                status="timeout",
                error=f"Tool timed out: {exc}",
                completed_at=utc_now(),
                duration_ms=int((time.monotonic() - started) * 1000),
            )
        except Exception as exc:  # noqa: BLE001 - isolate tool failures
            await context.event_bus.publish(
                _event(
                    context,
                    EventType.TOOL_COMPLETED,
                    payload={"tool": tool_name, "error": str(exc)},
                )
            )
            return ExecutionResult(
                tool_name=tool_name,
                status="failed",
                error=str(exc),
                completed_at=utc_now(),
                duration_ms=int((time.monotonic() - started) * 1000),
            )

        await context.event_bus.publish(
            _event(
                context,
                EventType.TOOL_COMPLETED,
                payload={"tool": tool_name, "success": result.success},
            )
        )
        return ExecutionResult(
            tool_name=tool_name,
            status="success" if result.success else "failed",
            output=result.output,
            error=result.error,
            completed_at=utc_now(),
            duration_ms=int((time.monotonic() - started) * 1000),
        )


def _event(
    context: LoopContext,
    event_type: EventType,
    *,
    agent_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> Event:
    return Event.create(
        event_type,
        task_id=context.task_id,
        agent_id=agent_id,
        session_id=context.session_id,
        payload={
            "loop_id": context.loop_id,
            "iteration": context.state.iteration_count,
            **(payload or {}),
        },
    )


def _cancelled_result(
    context: LoopContext, reason: str, *, agent_id: str | None = None
) -> ExecutionResult:
    return ExecutionResult(
        agent_id=agent_id,
        status="cancelled",
        error=reason,
        completed_at=utc_now(),
    )
