"""Research Agent: the second specialized agent (Phase 7).

The Research Agent is designed around the existing tool architecture. It does
NOT introduce unrestricted browser/shell execution. Its only research surface
is the :class:`~app.tools.impl.SearchTool` — a deterministic, LOW-risk
boundary registered through the :class:`ToolRegistry` and executed through the
:class:`ToolExecutor` (policy enforced). The agent never executes arbitrary
tools and never bypasses the registry/executor/policy.

Architecture::

    User request
        ↓
    ResearchAgent.execute()
        ↓
    query extraction (deterministic)
        ↓
    SearchTool via ToolExecutor (policy → registry → schema → execute)
        ↓
    structured research result (sources + summary)

The agent is honest: when no search source is configured, it reports that no
sources were available rather than fabricating results.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from app.agents.interface import (
    AgentCapability,
    AgentContext,
    AgentInfo,
    AgentInterface,
    AgentResult,
    AgentStatus,
)
from app.core.identifiers import generate_id
from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.tools.executor import ToolExecutor

logger = get_logger("agents.research")

__all__ = ["ResearchAgent", "RESEARCH_AGENT_ID"]

RESEARCH_AGENT_ID = "research.agent"
_MAX_INPUT_LENGTH = 1000


class ResearchAgent(AgentInterface):
    """Specialized agent for research tasks.

    Uses the :class:`SearchTool` via the :class:`ToolExecutor` for all
    information gathering. Never performs unrestricted browser/shell access.
    The search tool is the *only* research surface; it is resolved through the
    :class:`ToolRegistry` and gated by the :class:`ToolPolicy`.
    """

    def __init__(self, tool_executor: ToolExecutor | None = None) -> None:
        self._tool_executor = tool_executor
        self._info = AgentInfo(
            agent_id=RESEARCH_AGENT_ID,
            name="Research Agent",
            description=(
                "Performs research using the SearchTool boundary. Never uses "
                "unrestricted browser/shell access; all search goes through "
                "ToolRegistry/ToolExecutor/ToolPolicy."
            ),
            capabilities={AgentCapability.RESEARCH, AgentCapability.REASONING},
            version="0.1.0",
        )

    @property
    def info(self) -> AgentInfo:
        return self._info

    async def execute(self, context: AgentContext) -> AgentResult:
        user_input = context.input
        if user_input is None:
            return self._fail(context, "No input provided.")
        text = str(user_input).strip()
        if not text:
            return self._fail(context, "Empty input.")
        if len(text) > _MAX_INPUT_LENGTH:
            return self._fail(context, f"Input too long (max {_MAX_INPUT_LENGTH} chars).")

        query = _extract_query(text)
        if not query:
            return self._fail(context, "Could not identify a research query in the input.")

        if self._tool_executor is None:
            return self._fail(context, "No tool executor configured; cannot research.")

        from app.tools.executor import ToolCallRecord

        call = ToolCallRecord(
            tool_call_id=generate_id("call"),
            tool_name="search",
            arguments={"query": query, "limit": 5},
        )

        try:
            outcome = await self._tool_executor.execute_call(
                call,
                task_id=context.task_id,
                session_id=context.session_id,
            )
        except Exception as exc:  # noqa: BLE001 - never fabricate a result
            return self._fail(context, f"Tool execution failed: {exc}")

        if outcome.confirmation is not None:
            return AgentResult(
                task_id=context.task_id,
                agent_id=self.agent_id,
                status=AgentStatus.WAITING_FOR_CONFIRMATION,
                output=None,
                metadata={
                    "query": query,
                    "confirmation_id": outcome.confirmation.confirmation_id,
                    "confirmation_request": outcome.confirmation.model_dump(),
                },
            )

        if outcome.skipped or outcome.result is None or not outcome.result.success:
            error = (
                outcome.result.error
                if outcome.result and outcome.result.error
                else outcome.skipped_reason or "Search tool failed."
            )
            return self._fail(context, error, query=query)

        tool_output = outcome.result.output
        results = tool_output.get("results", []) if isinstance(tool_output, dict) else []
        count = tool_output.get("count", len(results)) if isinstance(tool_output, dict) else 0
        summary = (
            f"Found {count} source(s) for '{query}'."
            if count
            else f"No sources available for '{query}'."
        )
        return AgentResult(
            task_id=context.task_id,
            agent_id=self.agent_id,
            status=AgentStatus.COMPLETED,
            output={
                "query": query,
                "sources": results,
                "count": count,
                "summary": summary,
            },
            metadata={
                "tool": "search",
                "verified": count > 0,
                "query": query,
            },
        )

    def _fail(self, context: AgentContext, error: str, *, query: str | None = None) -> AgentResult:
        logger.bind(agent_id=self.agent_id, event="research.failed", error=error).warning(
            "Research agent failure: {}", error
        )
        return AgentResult(
            task_id=context.task_id,
            agent_id=self.agent_id,
            status=AgentStatus.FAILED,
            output=None,
            error=error,
            metadata={"query": query} if query else {},
        )


def _extract_query(text: str) -> str | None:
    """Extract a research query from natural-language input.

    Strips leading question words / research verbs to isolate the topic.
    """
    cleaned = text.strip().rstrip("?.!")
    if not cleaned:
        return None
    # Remove leading research/question phrases (deterministic).
    cleaned = re.sub(
        r"^(?:research|find out|look up|search(?: for)?|investigate|tell me about|what(?:'s| is| are)|how (?:do|does|to)|why (?:is|do|does))\b\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    return cleaned or None
