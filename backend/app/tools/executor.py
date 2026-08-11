"""Tool executor: the controlled bridge between LLM/plan and tool execution.

The LLM (or the plan stage) may *request* a tool call, but the actual
execution always flows through the :class:`ToolExecutor`:

::

    LLM / Plan
        ↓
    ToolCall
        ↓
    ToolExecutor
        ↓ 1. resolve tool via ToolRegistry (no arbitrary imports)
        ↓ 2. validate arguments against the tool's input schema
        ↓ 3. evaluate the ToolPolicy (ALLOW / DENY / REQUIRE_CONFIRMATION)
        ↓ 4. check tool-call limits and repeated-call protection
        ↓ 5. execute the tool (with timeout)
        ↓ 6. normalize the result into a typed ToolResult + Observation

The executor never allows the LLM to bypass the registry or the policy. Tool
names outside the registry are rejected. Invalid arguments are rejected.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import anyio
from jsonschema import ValidationError as JSONSchemaValidationError
from jsonschema import validate as validate_schema

from app.core.logging import get_logger
from app.events import Event, EventBus, EventType
from app.tools.confirmation import ConfirmationRequest, ConfirmationStore
from app.tools.interface import ToolContext, ToolInfo, ToolInterface, ToolResult
from app.tools.policy import PolicyDecision, ToolPolicy
from app.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from app.runtime.loop.observation import Observation

logger = get_logger("tools.executor")

__all__ = [
    "ToolExecutor",
    "ToolExecutionError",
    "ToolCallRecord",
    "ToolExecutionOutcome",
]


def _observation_cls():
    """Lazy import to avoid a circular dependency with the loop package."""
    from app.runtime.loop.observation import Observation

    return Observation


class ToolExecutionError(Exception):
    """Raised when a tool call cannot be resolved or is rejected."""

    def __init__(self, message: str, *, tool_name: str | None = None, reason: str = "") -> None:
        super().__init__(message)
        self.tool_name = tool_name
        self.reason = reason


@dataclass(frozen=True)
class ToolCallRecord:
    """A normalized record of a tool call requested by the LLM/plan."""

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]


@dataclass
class ToolExecutionOutcome:
    """The full outcome of attempting to execute a tool call.

    ``result`` is ``None`` when execution was skipped (denied, confirmation
    required, or limits exceeded). ``observation`` is always present so the
    loop can reason about what happened.
    """

    result: ToolResult | None
    observation: Observation
    decision: PolicyDecision | None = None
    confirmation: ConfirmationRequest | None = None
    skipped: bool = False
    skipped_reason: str | None = None


@dataclass
class ToolExecutor:
    """Resolves, validates, policy-checks, and executes tool calls.

    Depends on the :class:`ToolRegistry`, :class:`ToolPolicy`, and an optional
    :class:`ConfirmationStore` + :class:`EventBus`. Never depends on a specific
    tool implementation.
    """

    registry: ToolRegistry
    policy: ToolPolicy
    confirmation_store: ConfirmationStore = field(default_factory=ConfirmationStore)
    event_bus: EventBus | None = None
    # Tracks repeated identical calls per task for loop protection.
    _call_history: dict[str, list[ToolCallRecord]] = field(default_factory=dict)

    async def execute_call(
        self,
        call: ToolCallRecord,
        *,
        task_id: str | None = None,
        session_id: str | None = None,
        iteration: int = 0,
        timeout_seconds: float = 60.0,
    ) -> ToolExecutionOutcome:
        """Execute a single tool call through the full controlled pipeline."""
        await self._publish(EventType.TOOL_CALL_REQUESTED, call, task_id, session_id, iteration)

        # 1. Resolve tool via the registry.
        if not self.registry.exists(call.tool_name):
            return self._skip(
                call,
                task_id,
                session_id,
                iteration,
                reason=f"Tool '{call.tool_name}' is not registered.",
            )
        tool = self.registry.get(call.tool_name)
        info = tool.info

        # 2. Validate arguments against the tool's input schema.
        if info.input_schema:
            try:
                validate_schema(call.arguments, info.input_schema)
            except JSONSchemaValidationError as exc:
                return self._skip(
                    call,
                    task_id,
                    session_id,
                    iteration,
                    reason=f"Invalid arguments for '{call.tool_name}': {exc.message}",
                )

        # 3. Evaluate the policy.
        decision = self.policy.evaluate(
            info,
            arguments=call.arguments,
            task_id=task_id,
            session_id=session_id,
        )
        if decision.denied:
            obs = _observation_cls().from_denial(
                tool_name=call.tool_name,
                tool_call_id=call.tool_call_id,
                reason=decision.reason,
                task_id=task_id,
                session_id=session_id,
                iteration=iteration,
            )
            await self._publish_observation(obs, task_id, session_id, iteration)
            await self._publish(
                EventType.TOOL_POLICY_DENIED,
                call,
                task_id,
                session_id,
                iteration,
                extra={"reason": decision.reason},
            )
            return ToolExecutionOutcome(
                result=None,
                observation=obs,
                decision=decision,
                skipped=True,
                skipped_reason=decision.reason,
            )

        if decision.requires_confirmation:
            confirm_req = ConfirmationRequest(
                tool_name=call.tool_name,
                arguments=call.arguments,
                risk_level=info.risk_level,
                task_id=task_id,
                session_id=session_id,
                loop_id=None,
                iteration=iteration,
                tool_call_id=call.tool_call_id,
                reason=decision.reason,
            )
            self.confirmation_store.create(confirm_req)
            obs = _observation_cls().from_confirmation_required(
                tool_name=call.tool_name,
                tool_call_id=call.tool_call_id,
                confirmation_id=confirm_req.confirmation_id,
                reason=decision.reason,
                task_id=task_id,
                session_id=session_id,
                iteration=iteration,
            )
            await self._publish_observation(obs, task_id, session_id, iteration)
            await self._publish(
                EventType.TOOL_CONFIRMATION_REQUIRED,
                call,
                task_id,
                session_id,
                iteration,
                extra={"confirmation_id": confirm_req.confirmation_id, "reason": decision.reason},
            )
            return ToolExecutionOutcome(
                result=None,
                observation=obs,
                decision=decision,
                confirmation=confirm_req,
                skipped=True,
                skipped_reason=decision.reason,
            )

        # 4. Execute the tool (with timeout).
        return await self._run_tool(
            tool,
            info,
            call,
            task_id=task_id,
            session_id=session_id,
            iteration=iteration,
            timeout_seconds=timeout_seconds,
        )

    async def execute_confirmed(
        self,
        confirmation_id: str,
        *,
        task_id: str | None = None,
        session_id: str | None = None,
        iteration: int = 0,
        timeout_seconds: float = 60.0,
    ) -> ToolExecutionOutcome:
        """Execute a tool call that was previously held for confirmation.

        The confirmation must be in APPROVED state. Rebuilds the call from the
        stored request and runs the tool directly (policy already satisfied).
        """
        req = self.confirmation_store.get(confirmation_id)
        if req is None:
            raise ToolExecutionError(
                f"Confirmation '{confirmation_id}' not found.", reason="not_found"
            )
        if not req.is_approved:
            raise ToolExecutionError(
                f"Confirmation '{confirmation_id}' is not approved (status={req.status.value}).",
                reason="not_approved",
            )
        if not self.registry.exists(req.tool_name):
            return self._skip(
                ToolCallRecord(
                    tool_call_id=confirmation_id,
                    tool_name=req.tool_name,
                    arguments=dict(req.arguments),
                ),
                task_id,
                session_id,
                iteration,
                reason=f"Tool '{req.tool_name}' is no longer registered.",
            )
        tool = self.registry.get(req.tool_name)
        call = ToolCallRecord(
            tool_call_id=confirmation_id,
            tool_name=req.tool_name,
            arguments=dict(req.arguments),
        )
        return await self._run_tool(
            tool,
            tool.info,
            call,
            task_id=task_id,
            session_id=session_id,
            iteration=iteration,
            timeout_seconds=timeout_seconds,
        )

    def record_call(self, task_id: str, call: ToolCallRecord) -> None:
        """Record a call for repeated-call detection."""
        self._call_history.setdefault(task_id, []).append(call)

    def repeated_call_count(self, task_id: str, call: ToolCallRecord) -> int:
        """Count how many identical calls have already been recorded for this task."""
        history = self._call_history.get(task_id, [])
        return sum(
            1 for c in history if c.tool_name == call.tool_name and c.arguments == call.arguments
        )

    # --- internals ---

    async def _run_tool(
        self,
        tool: ToolInterface,
        info: ToolInfo,
        call: ToolCallRecord,
        *,
        task_id: str | None,
        session_id: str | None,
        iteration: int,
        timeout_seconds: float,
    ) -> ToolExecutionOutcome:
        await self._publish(
            EventType.TOOL_CALL_STARTED,
            call,
            task_id,
            session_id,
            iteration,
        )
        ctx = ToolContext(
            tool_call_id=call.tool_call_id,
            task_id=task_id,
            session_id=session_id,
            arguments=dict(call.arguments),
            metadata={"iteration": iteration},
        )
        started = time.monotonic()
        try:
            with anyio.fail_after(timeout_seconds):
                result: ToolResult = await tool.execute(ctx)
        except TimeoutError as exc:
            result = ToolResult(
                invocation_id=ctx.invocation_id,
                tool_call_id=call.tool_call_id,
                name=call.tool_name,
                success=False,
                error=f"Tool timed out: {exc}",
                execution_time_ms=int((time.monotonic() - started) * 1000),
            )
        except Exception as exc:  # noqa: BLE001 - normalize all tool failures
            result = ToolResult(
                invocation_id=ctx.invocation_id,
                tool_call_id=call.tool_call_id,
                name=call.tool_name,
                success=False,
                error=str(exc),
                execution_time_ms=int((time.monotonic() - started) * 1000),
            )

        if result.execution_time_ms is None:
            result = result.model_copy(
                update={"execution_time_ms": int((time.monotonic() - started) * 1000)}
            )
        if result.tool_call_id is None:
            result = result.model_copy(update={"tool_call_id": call.tool_call_id})

        event_type = EventType.TOOL_CALL_COMPLETED if result.success else EventType.TOOL_CALL_FAILED
        await self._publish(
            event_type,
            call,
            task_id,
            session_id,
            iteration,
            extra={"success": result.success, "error": result.error},
        )

        obs = _observation_cls().from_tool_result(
            tool_name=call.tool_name,
            tool_call_id=call.tool_call_id,
            result=result.output,
            success=result.success,
            task_id=task_id,
            session_id=session_id,
            iteration=iteration,
            error=result.error,
        )
        await self._publish_observation(obs, task_id, session_id, iteration)
        return ToolExecutionOutcome(result=result, observation=obs)

    def _skip(
        self,
        call: ToolCallRecord,
        task_id: str | None,
        session_id: str | None,
        iteration: int,
        *,
        reason: str,
    ) -> ToolExecutionOutcome:
        obs = _observation_cls().from_tool_result(
            tool_name=call.tool_name,
            tool_call_id=call.tool_call_id,
            result=None,
            success=False,
            task_id=task_id,
            session_id=session_id,
            iteration=iteration,
            error=reason,
        )
        return ToolExecutionOutcome(
            result=None,
            observation=obs,
            skipped=True,
            skipped_reason=reason,
        )

    async def _publish(
        self,
        event_type: EventType,
        call: ToolCallRecord,
        task_id: str | None,
        session_id: str | None,
        iteration: int,
        *,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if self.event_bus is None:
            return
        payload = {
            "tool": call.tool_name,
            "tool_call_id": call.tool_call_id,
            "iteration": iteration,
            **(extra or {}),
        }
        await self.event_bus.publish(
            Event.create(
                event_type,
                task_id=task_id,
                session_id=session_id,
                payload=payload,
                metadata={"tool": call.tool_name, "iteration": iteration},
            )
        )

    async def _publish_observation(
        self, obs: Observation, task_id: str | None, session_id: str | None, iteration: int
    ) -> None:
        if self.event_bus is None:
            return
        await self.event_bus.publish(
            Event.create(
                EventType.OBSERVATION_CREATED,
                task_id=task_id,
                session_id=session_id,
                payload={
                    "source": obs.source,
                    "type": obs.type.value,
                    "success": obs.success,
                    "tool": obs.tool_name,
                    "iteration": iteration,
                },
                metadata={"iteration": iteration},
            )
        )
