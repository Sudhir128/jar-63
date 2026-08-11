"""Math Agent: the first specialized agent (Phase 4).

The Math Agent understands mathematical user requests, determines whether
calculation is required, and uses the :class:`CalculatorTool` for the actual
arithmetic — never ``eval()`` or ``exec()``. The LLM may reason about *how*
to solve a problem, but the computation is always performed by the safe tool
layer.

Architecture::

    User request
        ↓
    MathAgent.execute()
        ↓
    Expression extraction (deterministic)
        ↓
    CalculatorTool (safe AST evaluation)
        ↓
    AgentResult (numeric answer + expression)

The agent is registered through the :class:`AgentRegistry` and never bypasses
it. It depends on the :class:`ToolExecutor` (which enforces policy) to invoke
the calculator — it never calls the tool directly.
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

logger = get_logger("agents.math")

__all__ = ["MathAgent", "MATH_AGENT_ID"]

MATH_AGENT_ID = "math.agent"

# Matches arithmetic expressions: digits, operators, parentheses, decimal points.
# This is a greedy match that captures the full expression including parentheses.
_EXPR_PATTERN = re.compile(
    r"[-+/*%().]*\d[\d\s+\-*/%.()//]*",
)
_MAX_INPUT_LENGTH = 500


class MathAgent(AgentInterface):
    """Specialized agent for mathematical computation.

    Uses the :class:`CalculatorTool` via the :class:`ToolExecutor` for all
    arithmetic. No ``eval()`` or ``exec()`` is used anywhere. The agent
    handles invalid input, division by zero, oversized expressions, tool
    failures, and malformed tool calls gracefully — the user always receives
    a clear failure rather than a fabricated answer.
    """

    def __init__(self, tool_executor: ToolExecutor | None = None) -> None:
        self._tool_executor = tool_executor
        self._info = AgentInfo(
            agent_id=MATH_AGENT_ID,
            name="Math Agent",
            description=(
                "Performs arithmetic computation using the CalculatorTool. "
                "Supports +, -, *, /, //, %, **, parentheses, percentages, "
                "and basic arithmetic expressions. No eval/exec."
            ),
            capabilities={AgentCapability.MATH, AgentCapability.REASONING},
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

        expression = _extract_expression(text)
        if not expression:
            return self._fail(
                context,
                "Could not identify a mathematical expression in the input.",
            )

        if self._tool_executor is None:
            return self._fail(context, "No tool executor configured; cannot compute.")

        from app.tools.executor import ToolCallRecord

        call = ToolCallRecord(
            tool_call_id=generate_id("call"),
            tool_name="calculator",
            arguments={"expression": expression},
        )

        try:
            outcome = await self._tool_executor.execute_call(
                call,
                task_id=context.task_id,
                session_id=context.session_id,
            )
        except Exception as exc:  # noqa: BLE001 - never fabricate a result
            return self._fail(context, f"Tool execution failed: {exc}")

        # If the tool requires confirmation, signal the loop to pause.
        # The confirmation request is carried in metadata so the execute
        # stage can set it on the loop state.
        if outcome.confirmation is not None:
            return AgentResult(
                task_id=context.task_id,
                agent_id=self.agent_id,
                status=AgentStatus.WAITING_FOR_CONFIRMATION,
                output=None,
                metadata={
                    "expression": expression,
                    "confirmation_id": outcome.confirmation.confirmation_id,
                    "confirmation_request": outcome.confirmation.model_dump(),
                },
            )

        if outcome.skipped or outcome.result is None or not outcome.result.success:
            error = (
                outcome.result.error
                if outcome.result and outcome.result.error
                else outcome.skipped_reason or "Calculator tool failed."
            )
            return self._fail(context, error, expression=expression)

        result_value = outcome.result.output
        if isinstance(result_value, dict) and "result" in result_value:
            numeric_result = result_value["result"]
        else:
            numeric_result = result_value

        return AgentResult(
            task_id=context.task_id,
            agent_id=self.agent_id,
            status=AgentStatus.COMPLETED,
            output={
                "result": numeric_result,
                "expression": expression,
                "raw_input": text,
            },
            metadata={
                "tool": "calculator",
                "verified": True,
                "expression": expression,
            },
        )

    def _fail(
        self, context: AgentContext, error: str, *, expression: str | None = None
    ) -> AgentResult:
        logger.bind(agent_id=self.agent_id, event="math.failed", error=error).warning(
            "Math agent failure: {}", error
        )
        return AgentResult(
            task_id=context.task_id,
            agent_id=self.agent_id,
            status=AgentStatus.FAILED,
            output=None,
            error=error,
            metadata={"expression": expression} if expression else {},
        )


def _extract_expression(text: str) -> str | None:
    """Extract a mathematical expression from natural-language input.

    Handles patterns like:
    * "What is 238 * 47?"
    * "Calculate (25 + 15) * 3"
    * "25% of 800" → "800 * (25 / 100)"
    * "10 / 0"
    * "2 ** 10"
    """
    # Handle percentage patterns: "X% of Y" → "Y * (X / 100)"
    pct_match = re.search(r"(\d+\.?\d*)\s*%\s*of\s*(\d+\.?\d*)", text, re.IGNORECASE)
    if pct_match:
        pct_val = float(pct_match.group(1))
        base_val = float(pct_match.group(2))
        return f"{base_val} * ({pct_val} / 100)"

    # Handle standalone "X%" → "X / 100"
    # Negative lookahead prevents matching "17 % 5" (modulo) as a percentage.
    standalone_pct = re.search(r"(\d+\.?\d*)\s*%(?!\s*[\d.])", text)
    if standalone_pct and "of" not in text.lower():
        pct_val = float(standalone_pct.group(1))
        return f"{pct_val} / 100"

    # Find all arithmetic expression matches and pick the longest.
    matches = _EXPR_PATTERN.findall(text)
    if not matches:
        return None
    expression = max(matches, key=len).strip()
    # Normalize: collapse whitespace around operators.
    expression = re.sub(r"\s+", " ", expression).strip()
    # Remove trailing operators or incomplete fragments.
    expression = re.sub(r"[-+*/%.]+$", "", expression).strip()
    # Remove trailing opening parenthesis fragments.
    expression = re.sub(r"\($", "", expression).strip()
    if not expression:
        return None
    # Validate it contains at least one digit.
    if not re.search(r"\d", expression):
        return None
    return expression
