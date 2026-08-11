"""Tests for Phase 3 tools: Calculator, Time, Health, Echo."""

from __future__ import annotations

from app.tools.impl import CalculatorTool, EchoTool, HealthTool, TimeTool
from app.tools.interface import RiskLevel, ToolCategory, ToolContext

# --- CalculatorTool ---


async def test_calculator_addition() -> None:
    tool = CalculatorTool()
    ctx = ToolContext(arguments={"expression": "2 + 3"})
    result = await tool.execute(ctx)
    assert result.success
    assert result.output["result"] == 5


async def test_calculator_multiplication() -> None:
    tool = CalculatorTool()
    ctx = ToolContext(arguments={"expression": "238 * 47"})
    result = await tool.execute(ctx)
    assert result.success
    assert result.output["result"] == 11186


async def test_calculator_division() -> None:
    tool = CalculatorTool()
    ctx = ToolContext(arguments={"expression": "100 / 4"})
    result = await tool.execute(ctx)
    assert result.success
    assert result.output["result"] == 25.0


async def test_calculator_floor_div_and_mod() -> None:
    tool = CalculatorTool()
    ctx = ToolContext(arguments={"expression": "17 // 5"})
    result = await tool.execute(ctx)
    assert result.success
    assert result.output["result"] == 3
    ctx2 = ToolContext(arguments={"expression": "17 % 5"})
    result2 = await tool.execute(ctx2)
    assert result2.output["result"] == 2


async def test_calculator_power() -> None:
    tool = CalculatorTool()
    ctx = ToolContext(arguments={"expression": "2 ** 10"})
    result = await tool.execute(ctx)
    assert result.success
    assert result.output["result"] == 1024


async def test_calculator_parentheses() -> None:
    tool = CalculatorTool()
    ctx = ToolContext(arguments={"expression": "(2 + 3) * 4"})
    result = await tool.execute(ctx)
    assert result.success
    assert result.output["result"] == 20


async def test_calculator_unary_minus() -> None:
    tool = CalculatorTool()
    ctx = ToolContext(arguments={"expression": "-5 + 10"})
    result = await tool.execute(ctx)
    assert result.success
    assert result.output["result"] == 5


async def test_calculator_floats() -> None:
    tool = CalculatorTool()
    ctx = ToolContext(arguments={"expression": "3.5 * 2"})
    result = await tool.execute(ctx)
    assert result.success
    assert result.output["result"] == 7.0


async def test_calculator_division_by_zero_fails() -> None:
    tool = CalculatorTool()
    ctx = ToolContext(arguments={"expression": "1 / 0"})
    result = await tool.execute(ctx)
    assert not result.success
    assert "Division by zero" in result.error


async def test_calculator_invalid_expression_fails() -> None:
    tool = CalculatorTool()
    ctx = ToolContext(arguments={"expression": "abc"})
    result = await tool.execute(ctx)
    assert not result.success
    assert "syntax" in result.error.lower() or "unsupported" in result.error.lower()


async def test_calculator_no_eval_or_exec() -> None:
    """The calculator must not use eval() or exec()."""
    import ast
    import inspect

    import app.tools.impl as impl

    source = inspect.getsource(impl)
    # Walk the AST to check for Call nodes targeting eval/exec.
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            func = node.func
            name = ""
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            assert name not in ("eval", "exec"), f"CalculatorTool uses {name}()"


async def test_calculator_rejects_function_calls() -> None:
    tool = CalculatorTool()
    ctx = ToolContext(arguments={"expression": "__import__('os').system('ls')"})
    result = await tool.execute(ctx)
    assert not result.success


async def test_calculator_rejects_assignment() -> None:
    tool = CalculatorTool()
    ctx = ToolContext(arguments={"expression": "x = 5"})
    result = await tool.execute(ctx)
    assert not result.success


async def test_calculator_empty_expression() -> None:
    tool = CalculatorTool()
    ctx = ToolContext(arguments={"expression": ""})
    result = await tool.execute(ctx)
    assert not result.success


async def test_calculator_missing_expression() -> None:
    tool = CalculatorTool()
    ctx = ToolContext(arguments={})
    result = await tool.execute(ctx)
    assert not result.success


async def test_calculator_expression_too_long() -> None:
    tool = CalculatorTool()
    ctx = ToolContext(arguments={"expression": "1 + " * 150})
    result = await tool.execute(ctx)
    assert not result.success
    assert "too long" in result.error.lower()


async def test_calculator_risk_level_low() -> None:
    tool = CalculatorTool()
    assert tool.risk_level is RiskLevel.LOW
    assert tool.info.category is ToolCategory.CALCULATOR
    assert tool.info.input_schema["type"] == "object"


# --- TimeTool ---


async def test_time_returns_iso() -> None:
    tool = TimeTool()
    ctx = ToolContext(arguments={})
    result = await tool.execute(ctx)
    assert result.success
    assert "utc_iso" in result.output
    assert "utc_date" in result.output
    assert "utc_time" in result.output
    assert "T" in result.output["utc_iso"]


async def test_time_risk_level_low() -> None:
    tool = TimeTool()
    assert tool.risk_level is RiskLevel.LOW


# --- HealthTool ---


async def test_health_returns_status() -> None:
    tool = HealthTool()
    ctx = ToolContext(arguments={})
    result = await tool.execute(ctx)
    assert result.success
    assert result.output["status"] == "ok"
    assert "runtime" in result.output["components"]


async def test_health_with_checker() -> None:
    async def checker():
        return {"database": "ok", "redis": "ok"}

    tool = HealthTool(health_checker=checker)
    ctx = ToolContext(arguments={})
    result = await tool.execute(ctx)
    assert result.success
    assert result.output["components"]["database"] == "ok"


async def test_health_risk_level_low() -> None:
    tool = HealthTool()
    assert tool.risk_level is RiskLevel.LOW


# --- EchoTool ---


async def test_echo_returns_arguments() -> None:
    tool = EchoTool()
    ctx = ToolContext(arguments={"message": "hello", "count": 3})
    result = await tool.execute(ctx)
    assert result.success
    assert result.output["message"] == "hello"
    assert result.output["count"] == 3


async def test_echo_risk_level_low() -> None:
    tool = EchoTool()
    assert tool.risk_level is RiskLevel.LOW
