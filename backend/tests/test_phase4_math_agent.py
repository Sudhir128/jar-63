"""Phase 4 tests: MathAgent — the first specialized agent.

Tests cover:
* AgentInterface compliance (info, agent_id, capabilities).
* Correct arithmetic via CalculatorTool (no eval/exec).
* Expression extraction from natural language.
* Percentage handling.
* Parenthesized expressions.
* Division by zero → structured failure.
* Invalid input → structured failure.
* Tool failure → no fabricated answer.
* Confirmation pause when tool requires confirmation.
* Registration through AgentRegistry.
"""

from __future__ import annotations

from app.agents.interface import (
    AgentCapability,
    AgentContext,
    AgentStatus,
)
from app.agents.math import MATH_AGENT_ID, MathAgent
from app.agents.math.verifier import MathVerifier
from app.agents.registry import AgentRegistry
from app.events import InMemoryEventBus
from app.tools.confirmation import ConfirmationStore
from app.tools.executor import ToolExecutor
from app.tools.impl import CalculatorTool
from app.tools.interface import (
    RiskLevel,
    ToolCategory,
    ToolContext,
    ToolInfo,
    ToolInterface,
    ToolResult,
)
from app.tools.policy import AllowAllToolPolicy, DefaultToolPolicy
from app.tools.registry import ToolRegistry


def _make_executor(*, policy=None) -> ToolExecutor:
    registry = ToolRegistry()
    executor = ToolExecutor(
        registry=registry,
        policy=policy or AllowAllToolPolicy(),
        confirmation_store=ConfirmationStore(),
        event_bus=InMemoryEventBus(),
    )
    return executor


async def _setup_math_agent(*, policy=None) -> tuple[MathAgent, ToolExecutor]:
    executor = _make_executor(policy=policy)
    await executor.registry.register(CalculatorTool())
    agent = MathAgent(tool_executor=executor)
    return agent, executor


# --- Interface compliance ---


def test_math_agent_info_is_correct() -> None:
    agent = MathAgent()
    info = agent.info
    assert info.agent_id == MATH_AGENT_ID
    assert info.name == "Math Agent"
    assert AgentCapability.MATH in info.capabilities
    assert AgentCapability.REASONING in info.capabilities


def test_math_agent_id_property() -> None:
    agent = MathAgent()
    assert agent.agent_id == MATH_AGENT_ID


async def test_math_agent_registered_via_registry() -> None:
    registry = AgentRegistry()
    agent = MathAgent()
    await registry.register(agent)
    assert registry.exists(MATH_AGENT_ID)
    retrieved = registry.get(MATH_AGENT_ID)
    assert retrieved is agent
    listed = registry.list()
    assert MATH_AGENT_ID in [a.agent_id for a in listed]


# --- Correct computation ---


async def test_math_agent_computes_multiplication() -> None:
    agent, _ = await _setup_math_agent()
    result = await agent.execute(AgentContext(input="What is 238 * 47?"))
    assert result.status is AgentStatus.COMPLETED
    assert result.output["result"] == 11186
    assert result.output["expression"] == "238 * 47"


async def test_math_agent_computes_addition() -> None:
    agent, _ = await _setup_math_agent()
    result = await agent.execute(AgentContext(input="Calculate 15 + 27"))
    assert result.status is AgentStatus.COMPLETED
    assert result.output["result"] == 42


async def test_math_agent_computes_subtraction() -> None:
    agent, _ = await _setup_math_agent()
    result = await agent.execute(AgentContext(input="What is 100 - 45?"))
    assert result.status is AgentStatus.COMPLETED
    assert result.output["result"] == 55


async def test_math_agent_computes_division() -> None:
    agent, _ = await _setup_math_agent()
    result = await agent.execute(AgentContext(input="What is 100 / 4?"))
    assert result.status is AgentStatus.COMPLETED
    assert result.output["result"] == 25.0


async def test_math_agent_computes_power() -> None:
    agent, _ = await _setup_math_agent()
    result = await agent.execute(AgentContext(input="What is 2 ** 10?"))
    assert result.status is AgentStatus.COMPLETED
    assert result.output["result"] == 1024


async def test_math_agent_computes_modulo() -> None:
    agent, _ = await _setup_math_agent()
    result = await agent.execute(AgentContext(input="What is 17 % 5?"))
    assert result.status is AgentStatus.COMPLETED
    assert result.output["result"] == 2


async def test_math_agent_computes_floor_division() -> None:
    agent, _ = await _setup_math_agent()
    result = await agent.execute(AgentContext(input="What is 17 // 5?"))
    assert result.status is AgentStatus.COMPLETED
    assert result.output["result"] == 3


# --- Parentheses & percentages ---


async def test_math_agent_handles_parentheses() -> None:
    agent, _ = await _setup_math_agent()
    result = await agent.execute(AgentContext(input="What is (25 + 15) * 3?"))
    assert result.status is AgentStatus.COMPLETED
    assert result.output["result"] == 120


async def test_math_agent_handles_nested_parentheses() -> None:
    agent, _ = await _setup_math_agent()
    result = await agent.execute(AgentContext(input="Calculate ((2 + 3) * 4) - 1"))
    assert result.status is AgentStatus.COMPLETED
    assert result.output["result"] == 19


async def test_math_agent_handles_percentage_of() -> None:
    agent, _ = await _setup_math_agent()
    result = await agent.execute(AgentContext(input="What is 25% of 800?"))
    assert result.status is AgentStatus.COMPLETED
    assert result.output["result"] == 200.0


async def test_math_agent_handles_standalone_percentage() -> None:
    agent, _ = await _setup_math_agent()
    result = await agent.execute(AgentContext(input="What is 50%?"))
    assert result.status is AgentStatus.COMPLETED
    assert result.output["result"] == 0.5


async def test_math_agent_handles_decimals() -> None:
    agent, _ = await _setup_math_agent()
    result = await agent.execute(AgentContext(input="What is 3.5 * 2?"))
    assert result.status is AgentStatus.COMPLETED
    assert result.output["result"] == 7.0


# --- Error handling ---


async def test_math_agent_division_by_zero_fails() -> None:
    agent, _ = await _setup_math_agent()
    result = await agent.execute(AgentContext(input="What is 10 / 0?"))
    assert result.status is AgentStatus.FAILED
    assert result.output is None
    assert "division" in (result.error or "").lower() or "zero" in (result.error or "").lower()


async def test_math_agent_no_input_fails() -> None:
    agent, _ = await _setup_math_agent()
    result = await agent.execute(AgentContext(input=None))
    assert result.status is AgentStatus.FAILED
    assert "no input" in (result.error or "").lower()


async def test_math_agent_empty_input_fails() -> None:
    agent, _ = await _setup_math_agent()
    result = await agent.execute(AgentContext(input="   "))
    assert result.status is AgentStatus.FAILED
    assert "empty" in (result.error or "").lower()


async def test_math_agent_non_math_input_fails() -> None:
    agent, _ = await _setup_math_agent()
    result = await agent.execute(AgentContext(input="Hello, how are you?"))
    assert result.status is AgentStatus.FAILED
    assert "expression" in (result.error or "").lower() or "math" in (result.error or "").lower()


async def test_math_agent_oversized_input_fails() -> None:
    agent, _ = await _setup_math_agent()
    long_input = "1 + 1 " * 200
    result = await agent.execute(AgentContext(input=long_input))
    assert result.status is AgentStatus.FAILED
    assert "too long" in (result.error or "").lower()


async def test_math_agent_without_executor_fails_gracefully() -> None:
    agent = MathAgent(tool_executor=None)
    result = await agent.execute(AgentContext(input="What is 2 + 2?"))
    assert result.status is AgentStatus.FAILED
    assert "tool executor" in (result.error or "").lower()


# --- Tool failure handling ---


class _FailingCalculator(ToolInterface):
    """A calculator-shaped tool that always fails."""

    def __init__(self) -> None:
        self._info = ToolInfo(
            tool_id="calculator",
            name="calculator",
            description="Always-failing calculator for testing.",
            category=ToolCategory.CALCULATOR,
            risk_level=RiskLevel.LOW,
            input_schema={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        )

    @property
    def info(self) -> ToolInfo:
        return self._info

    async def execute(self, context: ToolContext) -> ToolResult:
        return ToolResult(
            invocation_id=context.invocation_id,
            tool_call_id=context.tool_call_id,
            name="calculator",
            success=False,
            error="Simulated calculator failure.",
        )


async def test_math_agent_tool_failure_no_fabricated_answer() -> None:
    registry = ToolRegistry()
    await registry.register(_FailingCalculator())
    executor = ToolExecutor(
        registry=registry,
        policy=AllowAllToolPolicy(),
        confirmation_store=ConfirmationStore(),
        event_bus=InMemoryEventBus(),
    )
    agent = MathAgent(tool_executor=executor)
    result = await agent.execute(AgentContext(input="What is 2 + 2?"))
    assert result.status is AgentStatus.FAILED
    assert result.output is None
    assert "simulated" in (result.error or "").lower()


# --- Confirmation pause ---


async def test_math_agent_signals_waiting_for_confirmation() -> None:
    """When the calculator requires confirmation, the agent returns
    WAITING_FOR_CONFIRMATION instead of failing."""
    registry = ToolRegistry()
    await registry.register(CalculatorTool())
    # Make calculator MEDIUM risk so confirmation is triggered.
    calc = registry.get("calculator")
    calc._info = calc.info.model_copy(update={"risk_level": RiskLevel.MEDIUM})
    executor = ToolExecutor(
        registry=registry,
        policy=DefaultToolPolicy(require_confirmation_for_medium=True),
        confirmation_store=ConfirmationStore(),
        event_bus=InMemoryEventBus(),
    )
    agent = MathAgent(tool_executor=executor)
    result = await agent.execute(AgentContext(input="What is 238 * 47?"))
    assert result.status is AgentStatus.WAITING_FOR_CONFIRMATION
    assert result.output is None
    assert "confirmation_id" in result.metadata
    assert "confirmation_request" in result.metadata
    assert result.metadata["expression"] == "238 * 47"


# --- Verifier ---


async def test_math_verifier_confirms_correct_result() -> None:
    verifier = MathVerifier()
    actual = {"expression": "238 * 47", "result": 11186}
    result = await verifier.verify(actual)
    assert result.passed is True


async def test_math_verifier_rejects_wrong_result() -> None:
    verifier = MathVerifier()
    actual = {"expression": "238 * 47", "result": 99999}
    result = await verifier.verify(actual)
    assert result.passed is False


async def test_math_verifier_handles_missing_fields() -> None:
    verifier = MathVerifier()
    result = await verifier.verify({"expression": "2+2"})
    assert result.passed is False


async def test_math_verifier_handles_non_dict() -> None:
    verifier = MathVerifier()
    result = await verifier.verify("not a dict")
    assert result.passed is False


async def test_math_verifier_handles_division_by_zero_expression() -> None:
    verifier = MathVerifier()
    actual = {"expression": "10 / 0", "result": 0}
    result = await verifier.verify(actual)
    assert result.passed is False


async def test_math_verifier_allows_int_float_equivalence() -> None:
    verifier = MathVerifier()
    actual = {"expression": "4 / 2", "result": 2}
    result = await verifier.verify(actual)
    assert result.passed is True
