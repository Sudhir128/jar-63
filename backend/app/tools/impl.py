"""Concrete Phase 3 tools.

Only safe, low-risk foundation tools are implemented here:

* :class:`CalculatorTool` — safe arithmetic via AST (no eval/exec).
* :class:`TimeTool` — current date/time.
* :class:`HealthTool` — runtime/database/redis/LLM health.
* :class:`EchoTool` — testing utility.

No high-risk tools (shell, unrestricted Python, filesystem writes, database
mutations, deployment) are introduced in Phase 3.
"""

from __future__ import annotations

import ast
import operator
from datetime import datetime
from typing import Any

from app.core.identifiers import utc_now
from app.tools.interface import (
    RiskLevel,
    ToolCategory,
    ToolContext,
    ToolInfo,
    ToolInterface,
    ToolResult,
)

__all__ = [
    "SearchTool",
    "CalculatorTool",
    "TimeTool",
    "HealthTool",
    "EchoTool",
    "DEFAULT_TOOLS",
]


# ---------------------------------------------------------------------------
# CalculatorTool
# ---------------------------------------------------------------------------

_MAX_EXPRESSION_LENGTH = 200
_MAX_AST_DEPTH = 15

_AST_BINOPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_AST_UNARYOPS: dict[type, Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class CalculatorTool(ToolInterface):
    """Safe arithmetic calculator.

    Evaluates basic arithmetic expressions using Python's ``ast`` module with
    a restricted node whitelist. It does NOT use ``eval()`` or ``exec()``.
    Supports ``+``, ``-``, ``*``, ``/``, ``//``, ``%``, ``**``, unary ``+``,
    unary ``-``, parentheses, and numeric literals (int / float).
    """

    def __init__(self) -> None:
        self._info = ToolInfo(
            tool_id="calculator",
            name="calculator",
            description=(
                "Evaluates a basic arithmetic expression "
                "(+ - * / // % **, parentheses, numbers). No eval/exec."
            ),
            category=ToolCategory.CALCULATOR,
            risk_level=RiskLevel.LOW,
            requires_network=False,
            requires_confirmation=False,
            input_schema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The arithmetic expression to evaluate.",
                    }
                },
                "required": ["expression"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "result": {"description": "The numeric result."},
                    "expression": {"type": "string"},
                },
            },
            capabilities=["arithmetic"],
        )

    @property
    def info(self) -> ToolInfo:
        return self._info

    async def execute(self, context: ToolContext) -> ToolResult:
        expr = context.arguments.get("expression")
        if not isinstance(expr, str) or not expr.strip():
            return ToolResult(
                invocation_id=context.invocation_id,
                tool_call_id=context.tool_call_id,
                name=self.name,
                success=False,
                error="Argument 'expression' must be a non-empty string.",
            )
        if len(expr) > _MAX_EXPRESSION_LENGTH:
            return _fail(
                context, self.name, f"Expression too long (max {_MAX_EXPRESSION_LENGTH} chars)."
            )
        try:
            value = _safe_eval(expr)
        except _CalcError as exc:
            return _fail(context, self.name, str(exc))
        except Exception as exc:  # noqa: BLE001
            return _fail(context, self.name, f"Could not evaluate expression: {exc}")
        return ToolResult(
            invocation_id=context.invocation_id,
            tool_call_id=context.tool_call_id,
            name=self.name,
            success=True,
            output={"result": value, "expression": expr},
        )


class _CalcError(Exception):
    """Internal calculator error."""


def _safe_eval(expression: str) -> int | float:
    """Parse and evaluate an arithmetic expression safely via AST."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise _CalcError(f"Invalid syntax: {exc.msg}") from exc
    if _ast_depth(tree) > _MAX_AST_DEPTH:
        raise _CalcError(f"Expression too complex (max depth {_MAX_AST_DEPTH}).")
    return _eval_node(tree.body)


def _ast_depth(node: ast.AST) -> int:
    children = list(ast.iter_child_nodes(node))
    if not children:
        return 1
    return 1 + max(_ast_depth(c) for c in children)


def _eval_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, int | float):
            return node.value
        raise _CalcError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op_fn = _AST_BINOPS.get(type(node.op))
        if op_fn is None:
            raise _CalcError(f"Unsupported operator: {type(node.op).__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        try:
            return op_fn(left, right)
        except ZeroDivisionError as exc:
            raise _CalcError("Division by zero.") from exc
    if isinstance(node, ast.UnaryOp):
        op_fn = _AST_UNARYOPS.get(type(node.op))
        if op_fn is None:
            raise _CalcError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_fn(_eval_node(node.operand))
    raise _CalcError(f"Unsupported expression element: {type(node).__name__}")


# ---------------------------------------------------------------------------
# TimeTool
# ---------------------------------------------------------------------------


class TimeTool(ToolInterface):
    """Returns the current UTC date and time."""

    def __init__(self) -> None:
        self._info = ToolInfo(
            tool_id="time",
            name="time",
            description="Returns the current UTC date and time (ISO 8601).",
            category=ToolCategory.TIME,
            risk_level=RiskLevel.LOW,
            input_schema={"type": "object", "properties": {}},
            capabilities=["time"],
        )

    @property
    def info(self) -> ToolInfo:
        return self._info

    async def execute(self, context: ToolContext) -> ToolResult:
        now: datetime = utc_now()
        return ToolResult(
            invocation_id=context.invocation_id,
            tool_call_id=context.tool_call_id,
            name=self.name,
            success=True,
            output={
                "utc_iso": now.isoformat(),
                "utc_date": now.strftime("%Y-%m-%d"),
                "utc_time": now.strftime("%H:%M:%S"),
            },
        )


# ---------------------------------------------------------------------------
# HealthTool
# ---------------------------------------------------------------------------


class HealthTool(ToolInterface):
    """Returns runtime, database, Redis, and LLM/provider health.

    The tool accepts an optional callable for async dependency checks so it
    can be wired to the real runtime without hard coupling.
    """

    def __init__(
        self,
        *,
        health_checker: Any | None = None,
    ) -> None:
        self._health_checker = health_checker
        self._info = ToolInfo(
            tool_id="health",
            name="health",
            description="Returns health status of runtime, database, redis, and LLM providers.",
            category=ToolCategory.HEALTH,
            risk_level=RiskLevel.LOW,
            input_schema={"type": "object", "properties": {}},
            capabilities=["health"],
        )

    @property
    def info(self) -> ToolInfo:
        return self._info

    async def execute(self, context: ToolContext) -> ToolResult:
        components: dict[str, Any] = {"runtime": "ok"}
        if self._health_checker is not None:
            try:
                result = self._health_checker()
                if hasattr(result, "__await__"):
                    result = await result  # type: ignore[func-returns-value]
                if isinstance(result, dict):
                    components = result
            except Exception as exc:  # noqa: BLE001
                components["health_check_error"] = str(exc)
        return ToolResult(
            invocation_id=context.invocation_id,
            tool_call_id=context.tool_call_id,
            name=self.name,
            success=True,
            output={"components": components, "status": "ok"},
        )


# ---------------------------------------------------------------------------
# EchoTool
# ---------------------------------------------------------------------------


class EchoTool(ToolInterface):
    """Echoes its arguments — used for testing/demos."""

    def __init__(self) -> None:
        self._info = ToolInfo(
            tool_id="echo",
            name="echo",
            description="Echoes the provided arguments back (testing utility).",
            category=ToolCategory.CUSTOM,
            risk_level=RiskLevel.LOW,
            input_schema={"type": "object"},
            capabilities=["echo"],
        )

    @property
    def info(self) -> ToolInfo:
        return self._info

    async def execute(self, context: ToolContext) -> ToolResult:
        return ToolResult(
            invocation_id=context.invocation_id,
            tool_call_id=context.tool_call_id,
            name=self.name,
            success=True,
            output=dict(context.arguments),
        )


# ---------------------------------------------------------------------------
# SearchTool (Phase 7 research boundary)
# ---------------------------------------------------------------------------


class SearchTool(ToolInterface):
    """Deterministic search tool — the Phase 7 research boundary.

    This is a *clean tool boundary* for the ResearchAgent. It does NOT perform
    unrestricted browser/shell/network crawling. It accepts a query and an
    optional results source (a callable returning a list of deterministic
    hits). When no source is provided it returns an empty result set — the
    ResearchAgent then reports that no sources were available, rather than
    fabricating results.

    The tool is LOW risk and registered through the :class:`ToolRegistry`,
    executed through the :class:`ToolExecutor` (policy enforced). It is the
    only search surface the ResearchAgent may use.
    """

    def __init__(
        self,
        *,
        results_source: Any | None = None,
    ) -> None:
        self._results_source = results_source
        self._info = ToolInfo(
            tool_id="search",
            name="search",
            description=(
                "Deterministic search boundary for research agents. Returns "
                "structured results without unrestricted browser/shell access."
            ),
            category=ToolCategory.SEARCH,
            risk_level=RiskLevel.LOW,
            requires_network=False,
            requires_confirmation=False,
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results.",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "results": {"type": "array"},
                    "count": {"type": "integer"},
                    "query": {"type": "string"},
                },
            },
            capabilities=["research", "search"],
        )

    @property
    def info(self) -> ToolInfo:
        return self._info

    async def execute(self, context: ToolContext) -> ToolResult:
        query = context.arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            return ToolResult(
                invocation_id=context.invocation_id,
                tool_call_id=context.tool_call_id,
                name=self.name,
                success=False,
                error="Argument 'query' must be a non-empty string.",
            )
        limit = context.arguments.get("limit", 5)
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 5
        results: list[Any] = []
        if self._results_source is not None:
            try:
                raw = self._results_source(query)
                if hasattr(raw, "__await__"):
                    raw = await raw  # type: ignore[func-returns-value]
                if isinstance(raw, list):
                    results = raw[:limit]
            except Exception as exc:  # noqa: BLE001
                return ToolResult(
                    invocation_id=context.invocation_id,
                    tool_call_id=context.tool_call_id,
                    name=self.name,
                    success=False,
                    error=f"Search source error: {exc}",
                )
        return ToolResult(
            invocation_id=context.invocation_id,
            tool_call_id=context.tool_call_id,
            name=self.name,
            success=True,
            output={"results": results, "count": len(results), "query": query},
        )


def _fail(context: ToolContext, name: str, error: str) -> ToolResult:
    return ToolResult(
        invocation_id=context.invocation_id,
        tool_call_id=context.tool_call_id,
        name=name,
        success=False,
        error=error,
    )


def DEFAULT_TOOLS() -> list[ToolInterface]:
    """Construct the default set of Phase 3 foundation tools."""
    return [CalculatorTool(), TimeTool(), HealthTool(), EchoTool()]
