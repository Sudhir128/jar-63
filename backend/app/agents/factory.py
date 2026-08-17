"""Agent factory: construct a runtime instance from a definition (Phase 7).

The :class:`AgentFactory` maps a persistent :class:`AgentDefinition` to a
runnable :class:`AgentInterface` instance. Built-in kinds:

* ``MATH``    → :class:`~app.agents.math.MathAgent`
* ``RESEARCH`` → :class:`~app.agents.research.ResearchAgent`
* ``GENERIC`` → :class:`GenericAgent` (follows definition instructions and
  executes tools *only* through the :class:`ToolExecutor`)

The factory never instantiates agents from arbitrary modules — it is the
single place that knows how to build a runtime instance for a kind. Dynamic
agents are always ``GENERIC``: their definition's instructions guide behavior,
but they cannot execute arbitrary code; any tool they use must already be
registered in the :class:`ToolRegistry` and is gated by the
:class:`ToolPolicy`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.agents.definitions import AgentDefinition, AgentKind
from app.agents.interface import (
    AgentCapability,
    AgentContext,
    AgentInfo,
    AgentInterface,
    AgentResult,
    AgentStatus,
)
from app.agents.math import MATH_AGENT_ID, MathAgent
from app.agents.research import RESEARCH_AGENT_ID, ResearchAgent
from app.core.exceptions import AgentDefinitionValidationError
from app.core.identifiers import generate_id
from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.tools.executor import ToolExecutor

logger = get_logger("agents.factory")

__all__ = ["AgentFactory", "GenericAgent", "GENERIC_AGENT_PREFIX"]


GENERIC_AGENT_PREFIX = "dynamic.agent"


class GenericAgent(AgentInterface):
    """Tool-driven agent for dynamic/generic definitions.

    It executes *only* through the :class:`ToolExecutor`: it issues tool calls
    named in the definition's ``required_tools``/``preferred_tools`` (validated
    and policy-gated by the executor) and assembles a structured result. It
    never imports modules, never runs ``eval``/``exec``, never shells out, and
    never bypasses the registry/executor/policy.

    When no tools are declared, the agent returns its instructions as a
    structured "no-op capability" result so the loop can verify and terminate
    honestly.
    """

    def __init__(
        self,
        definition: AgentDefinition,
        *,
        tool_executor: ToolExecutor | None = None,
    ) -> None:
        self._definition = definition
        self._tool_executor = tool_executor
        self._info = AgentInfo(
            agent_id=definition.agent_id,
            name=definition.name,
            description=definition.description or definition.purpose,
            capabilities=set(definition.capabilities) | {AgentCapability.REASONING},
            version=definition.version,
        )

    @property
    def info(self) -> AgentInfo:
        return self._info

    async def execute(self, context: AgentContext) -> AgentResult:
        if self._tool_executor is None or not self._definition.required_tools:
            return AgentResult(
                task_id=context.task_id,
                agent_id=self.agent_id,
                status=AgentStatus.COMPLETED,
                output={
                    "kind": self._definition.kind.value,
                    "instructions": self._definition.instructions,
                    "tools_used": [],
                    "note": "No executable tools declared; returning structured no-op result.",
                },
                metadata={
                    "agent_version": self._definition.version,
                    "dynamic": self._definition.dynamic,
                    "verified": True,
                },
            )

        from app.tools.executor import ToolCallRecord

        user_input = str(context.input) if context.input is not None else ""
        arguments = {"input": user_input} if user_input else {}
        observations: list[dict] = []
        for tool_name in self._definition.required_tools:
            call = ToolCallRecord(
                tool_call_id=generate_id("call"),
                tool_name=tool_name,
                arguments=dict(arguments),
            )
            try:
                outcome = await self._tool_executor.execute_call(
                    call,
                    task_id=context.task_id,
                    session_id=context.session_id,
                )
            except Exception as exc:  # noqa: BLE001 - never fabricate; record and continue
                observations.append({"tool": tool_name, "success": False, "error": str(exc)})
                continue
            if outcome.confirmation is not None:
                return AgentResult(
                    task_id=context.task_id,
                    agent_id=self.agent_id,
                    status=AgentStatus.WAITING_FOR_CONFIRMATION,
                    output=None,
                    metadata={
                        "confirmation_id": outcome.confirmation.confirmation_id,
                        "confirmation_request": outcome.confirmation.model_dump(),
                        "tool": tool_name,
                    },
                )
            ok = outcome.result is not None and outcome.result.success
            observations.append(
                {
                    "tool": tool_name,
                    "success": ok,
                    "output": outcome.result.output if outcome.result else None,
                    "error": outcome.result.error if outcome.result else outcome.skipped_reason,
                }
            )
        success = all(o.get("success") for o in observations) if observations else True
        return AgentResult(
            task_id=context.task_id,
            agent_id=self.agent_id,
            status=AgentStatus.COMPLETED if success else AgentStatus.FAILED,
            output={
                "kind": self._definition.kind.value,
                "instructions": self._definition.instructions,
                "tools_used": [o["tool"] for o in observations],
                "observations": observations,
            },
            error=None if success else "One or more required tools failed.",
            metadata={
                "agent_version": self._definition.version,
                "dynamic": self._definition.dynamic,
                "tool_count": len(observations),
                "verified": success,
            },
        )


class AgentFactory:
    """Builds runtime :class:`AgentInterface` instances from definitions.

    The factory is the single place that knows how to construct a runtime
    instance for a given :class:`AgentKind`. Other modules must not
    instantiate agents directly — they go through the factory so the
    dispatcher can wire the :class:`ToolExecutor` consistently.
    """

    def __init__(self, *, tool_executor: ToolExecutor | None = None) -> None:
        self._tool_executor = tool_executor

    def build(self, definition: AgentDefinition) -> AgentInterface:
        kind = definition.kind
        if kind is AgentKind.MATH:
            return MathAgent(tool_executor=self._tool_executor)
        if kind is AgentKind.RESEARCH:
            return ResearchAgent(tool_executor=self._tool_executor)
        if kind is AgentKind.GENERIC:
            return GenericAgent(definition, tool_executor=self._tool_executor)
        raise AgentDefinitionValidationError(f"Unknown agent kind: {kind}")


def default_definitions() -> list[AgentDefinition]:
    """Return the built-in (non-dynamic) agent definitions.

    Registered into the catalog at startup so the router has agents to select.
    """
    from app.agents.definitions import AgentLifecycleState
    from app.llm.models import PrivacyLevel
    from app.tools.interface import RiskLevel

    return [
        AgentDefinition(
            agent_id=MATH_AGENT_ID,
            name="Math Agent",
            description="Performs arithmetic computation via the CalculatorTool.",
            purpose="Solve arithmetic and math expressions.",
            kind=AgentKind.MATH,
            capabilities={AgentCapability.MATH, AgentCapability.REASONING},
            task_types=["math", "arithmetic", "calculation"],
            required_tools=["calculator"],
            preferred_tools=["calculator"],
            verification_strategy="recompute independently and compare",
            model_requirements=["chat", "reasoning"],
            privacy=PrivacyLevel.INTERNAL,
            risk=RiskLevel.LOW,
            lifecycle=AgentLifecycleState.ACTIVE,
            version="0.1.0",
            dynamic=False,
        ),
        AgentDefinition(
            agent_id=RESEARCH_AGENT_ID,
            name="Research Agent",
            description="Performs research via the SearchTool boundary.",
            purpose="Gather structured research results for a query.",
            kind=AgentKind.RESEARCH,
            capabilities={AgentCapability.RESEARCH, AgentCapability.REASONING},
            task_types=["research", "search", "information"],
            required_tools=["search"],
            preferred_tools=["search"],
            verification_strategy="require at least one non-empty source",
            model_requirements=["chat", "reasoning"],
            privacy=PrivacyLevel.INTERNAL,
            risk=RiskLevel.LOW,
            lifecycle=AgentLifecycleState.ACTIVE,
            version="0.1.0",
            dynamic=False,
        ),
    ]
