"""LLM-backed planner.

Wraps an :class:`LLMClient` (selected via the :class:`ModelRouter`) to produce
a validated :class:`LLMPlan`, then translates it into the loop engine's
:class:`PlanResult` / :class:`NextAction`. On any LLM failure it falls back to
the deterministic planner and records the fallback decision.

The LLM **only** produces a structured plan. It never executes shell commands,
Python, browser actions, filesystem changes, or database writes. Execution
remains the Loop Engine's responsibility.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from app.config import LLMSettings
from app.core.logging import get_logger
from app.events import Event, EventBus, EventType
from app.llm.client import LLMClient
from app.llm.errors import InvalidStructuredOutputError, LLMError
from app.llm.models import (
    LLMMessage,
    LLMRequest,
    MessageRole,
    ModelCapability,
    PrivacyLevel,
    StructuredOutputSpec,
)
from app.llm.plan_schema import LLMPlan, LLMPlanStep
from app.llm.router import ModelRouter, ModelSelection, RoutingRequest
from app.runtime.loop.loop_context import LoopContext
from app.runtime.loop.loop_state import ActionType, NextAction, PlanStep
from app.runtime.loop.stages.plan import DefaultPlanStage, PlanResult

logger = get_logger("llm.planner")

__all__ = ["LLMPlanner", "PlannerFallbackReason"]


class PlannerFallbackReason:
    """Reasons the planner fell back to the deterministic planner."""

    LLM_DISABLED = "llm_disabled"
    NO_MODEL = "no_model"
    LLM_ERROR = "llm_error"
    INVALID_OUTPUT = "invalid_output"
    EMPTY_PLAN = "empty_plan"


class LLMPlanner:
    """Plans the next action using an LLM, with deterministic fallback."""

    def __init__(
        self,
        *,
        router: ModelRouter,
        settings: LLMSettings,
        fallback: DefaultPlanStage | None = None,
        event_bus: EventBus | None = None,
        client_override: LLMClient | None = None,
    ) -> None:
        self._router = router
        self._settings = settings
        self._fallback = fallback or DefaultPlanStage()
        self._event_bus = event_bus
        # Tests / DI may inject a specific client, bypassing routing.
        self._client_override = client_override

    async def plan(self, context: LoopContext) -> PlanResult:
        """Produce a plan, preferring the LLM and falling back deterministically."""
        if not self._settings.enabled:
            return await self._fallback_and_record(context, PlannerFallbackReason.LLM_DISABLED)

        try:
            selection = await self._select_model(context)
        except LLMError as exc:
            return await self._fallback_and_record(
                context, PlannerFallbackReason.NO_MODEL, detail=str(exc)
            )

        client = self._client_override or await self._client_for(selection)
        if client is None:
            return await self._fallback_and_record(
                context, PlannerFallbackReason.NO_MODEL, detail="no LLM client available"
            )

        try:
            llm_plan = await self._generate_plan(context, client, selection)
        except InvalidStructuredOutputError as exc:
            return await self._fallback_and_record(
                context, PlannerFallbackReason.INVALID_OUTPUT, detail=str(exc)
            )
        except ValidationError as exc:
            return await self._fallback_and_record(
                context, PlannerFallbackReason.INVALID_OUTPUT, detail=str(exc)
            )
        except LLMError as exc:
            return await self._fallback_and_record(
                context, PlannerFallbackReason.LLM_ERROR, detail=str(exc)
            )

        if llm_plan is None or not llm_plan.steps:
            return await self._fallback_and_record(context, PlannerFallbackReason.EMPTY_PLAN)

        return self._translate(llm_plan, selection, context)

    # --- model selection ---

    async def _select_model(self, context: LoopContext) -> ModelSelection:
        request = RoutingRequest(
            capabilities={ModelCapability.CHAT, ModelCapability.CODING},
            privacy=PrivacyLevel.INTERNAL,
            prefer_local=True,
            metadata={"stage": "plan", "loop_id": context.loop_id},
        )
        return await self._router.select(request)

    async def _client_for(self, selection: ModelSelection) -> LLMClient | None:
        providers = self._router._providers  # noqa: SLF001 - registry access for client lookup
        if not providers.exists(selection.provider):
            return None
        return providers.get(selection.provider)

    # --- generation ---

    async def _generate_plan(
        self,
        context: LoopContext,
        client: LLMClient,
        selection: ModelSelection,
    ) -> LLMPlan:
        messages = self._build_prompt(context, selection)
        spec = StructuredOutputSpec(name="plan", response_model=LLMPlan)
        request = LLMRequest(
            model=selection.model_id,
            messages=messages,
            temperature=0.2,
            structured_output=spec,
            privacy=PrivacyLevel.INTERNAL,
            metadata={"stage": "plan", "loop_id": context.loop_id},
        )
        response = await client.generate_structured(request, spec)
        parsed = response.metadata.get("parsed")
        if parsed is None:
            raise InvalidStructuredOutputError("LLM returned no parsed plan")
        return LLMPlan.model_validate(parsed)

    def _build_prompt(self, context: LoopContext, selection: ModelSelection) -> list[LLMMessage]:
        discovery = context.task.metadata.get("discovery") or {}
        agents = discovery.get("available_agents") or []
        tools = discovery.get("available_tools") or []
        goal = context.state.goal
        criteria = context.state.success_criteria
        return [
            LLMMessage(
                role=MessageRole.SYSTEM,
                content=(
                    "You are a planning module for the JAR-63 loop engine. "
                    "Produce a STRICT structured plan in JSON matching the given schema. "
                    "You MUST NOT execute anything — only describe the plan. "
                    "Each step must declare the capability it needs. When a tool is "
                    "available that can satisfy the step, set the 'tool' field to the "
                    "tool name and 'tool_arguments' to the arguments dict. When an "
                    "agent should perform the step, set 'agent_id'. Always include at "
                    "least one success criterion."
                ),
            ),
            LLMMessage(
                role=MessageRole.USER,
                content=(
                    f"Goal: {goal}\n"
                    f"Success criteria: {criteria}\n"
                    f"Available agents: {agents}\n"
                    f"Available tools: {tools}\n"
                    f"Produce a plan to achieve the goal."
                ),
            ),
        ]

    # --- translation ---

    def _translate(
        self,
        llm_plan: LLMPlan,
        selection: ModelSelection,
        context: LoopContext,
    ) -> PlanResult:
        """Translate a validated LLMPlan into the loop engine's PlanResult."""
        discovery = context.task.metadata.get("discovery") or {}
        registry = context.agent_registry
        tool_registry = context.tool_registry

        steps: list[PlanStep] = []
        first_step: PlanStep | None = None
        next_action = NextAction(action_type=ActionType.NONE, description="No actionable step.")
        first = llm_plan.first_step

        if first is not None:
            # Tool step: validate the tool exists in the registry.
            if first.is_tool_step and first.tool:
                if tool_registry.exists(first.tool):
                    action_type = ActionType.EXECUTE_TOOL
                    target = first.tool
                    desc = first.description
                    params: dict[str, Any] = {
                        "arguments": dict(first.tool_arguments),
                        "llm_step_id": first.step_id,
                    }
                else:
                    # Unknown tool — do not execute. Fall back to no-op.
                    action_type = ActionType.NONE
                    target = None
                    desc = f"LLM plan referenced unknown tool '{first.tool}'; no execution."
                    params = {"llm_step_id": first.step_id}
            else:
                agent_id = first.agent_id or discovery.get("agent_id")
                # Resolve: only execute a known agent. Unknown agents -> no-op.
                if agent_id and registry.exists(agent_id):
                    action_type = ActionType.EXECUTE_AGENT
                    target = agent_id
                    desc = first.description
                    params = {"llm_step_id": first.step_id}
                else:
                    # If the plan references an unknown agent, do not execute it.
                    agent_id = None
                    action_type = ActionType.NONE
                    target = None
                    desc = "LLM plan referenced an unknown agent; no execution."
                    params = {"llm_step_id": first.step_id}
            step = PlanStep(
                description=desc,
                action_type=action_type,
                target=target,
                parameters=params,
            )
            steps.append(step)
            first_step = step
            next_action = NextAction(
                action_type=action_type,
                target=target,
                description=desc,
                parameters=params,
                reasoning=f"LLM plan (model={selection.display}): {first.description}",
            )

        return PlanResult(
            steps=steps,
            current_step=first_step,
            next_action=next_action,
            reasoning_metadata={
                "planner": "llm",
                "model": selection.model_id,
                "provider": selection.provider,
                "plan_id": llm_plan.plan_id,
                "assumptions": llm_plan.assumptions,
                "success_criteria": llm_plan.success_criteria,
            },
        )

    # --- fallback ---

    async def _fallback_and_record(
        self,
        context: LoopContext,
        reason: str,
        *,
        detail: str | None = None,
    ) -> PlanResult:
        logger.bind(event="planner.fallback", reason=reason).info(
            "Falling back to deterministic planner: {}{}", reason, f" ({detail})" if detail else ""
        )
        await self._publish_fallback(context, reason, detail)
        result = await self._fallback.plan(context)
        # Tag the result so callers know a fallback occurred.
        result.reasoning_metadata["planner"] = "deterministic_fallback"
        result.reasoning_metadata["fallback_reason"] = reason
        if detail:
            result.reasoning_metadata["fallback_detail"] = detail
        return result

    async def _publish_fallback(
        self, context: LoopContext, reason: str, detail: str | None
    ) -> None:
        if self._event_bus is None:
            return
        await self._event_bus.publish(
            Event.create(
                EventType.MODEL_FALLBACK,
                task_id=context.task_id,
                session_id=context.session_id,
                payload={
                    "reason": reason,
                    "detail": detail,
                    "stage": "plan",
                    "loop_id": context.loop_id,
                },
            )
        )


def llm_plan_step_to_plan_step(step: LLMPlanStep, agent_id: str | None) -> PlanStep:
    """Convert an LLM plan step into a loop engine plan step (utility)."""
    return PlanStep(
        description=step.description,
        action_type=ActionType.EXECUTE_AGENT if agent_id else ActionType.NONE,
        target=agent_id,
        parameters={"llm_step_id": step.step_id},
    )


def _safe_metadata(**kwargs: Any) -> dict[str, Any]:
    return dict(kwargs)
