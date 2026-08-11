"""Phase 4 tests: confirmation resume lifecycle.

Tests cover:
* Confirmation store: create, approve, reject, expire, list_pending.
* Confirmation store: double-approve/reject is idempotent.
* LoopController: pause on confirmation, resume after approval.
* LoopController: resume after rejection (tool not executed).
* LoopService: list_pending_confirmations, approve_confirmation,
  reject_confirmation, resume_loop, resume_loop_after_rejection.
* Confirmation events emitted.
* Expiration handling.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from app.agents.math import MATH_AGENT_ID, MathAgent
from app.agents.math.verifier import MathVerifier
from app.agents.registry import AgentRegistry
from app.core.identifiers import utc_now
from app.events import Event, EventType, InMemoryEventBus
from app.runtime.loop.loop_context import LoopContext
from app.runtime.loop.loop_controller import LoopController
from app.runtime.loop.loop_policy import LoopPolicy
from app.runtime.loop.loop_result import LoopFinalStatus
from app.runtime.loop.loop_state import LoopState
from app.runtime.loop.service import LoopService
from app.runtime.loop.stages import (
    DefaultDiscoverStage,
    DefaultIterateStage,
    DefaultPlanStage,
    DefaultVerifyStage,
)
from app.runtime.loop.stages.execute import DefaultExecuteStage
from app.runtime.models import Task
from app.runtime.session_manager import SessionManager
from app.runtime.task_manager import TaskManager
from app.tools.confirmation import (
    DEFAULT_CONFIRMATION_TTL_SECONDS,
    ConfirmationRequest,
    ConfirmationStatus,
    ConfirmationStore,
)
from app.tools.executor import ToolExecutor
from app.tools.impl import CalculatorTool
from app.tools.interface import RiskLevel
from app.tools.policy import DefaultToolPolicy
from app.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# ConfirmationStore unit tests
# ---------------------------------------------------------------------------


def _make_request(**overrides) -> ConfirmationRequest:
    base = {
        "tool_name": "calculator",
        "arguments": {"expression": "2+2"},
        "risk_level": RiskLevel.MEDIUM,
        "task_id": "task_1",
        "session_id": "sess_1",
        "loop_id": "loop_1",
        "iteration": 0,
        "tool_call_id": "call_1",
        "reason": "MEDIUM risk tool",
    }
    base.update(overrides)
    return ConfirmationRequest(**base)


def test_store_create_stores_request() -> None:
    store = ConfirmationStore()
    req = _make_request()
    stored = store.create(req)
    assert stored.confirmation_id == req.confirmation_id
    assert store.get(req.confirmation_id) is not None
    assert len(store) == 1


def test_store_create_sets_default_ttl() -> None:
    store = ConfirmationStore()
    req = _make_request()
    stored = store.create(req)
    assert stored.expires_at is not None
    delta = stored.expires_at - stored.created_at
    assert abs(delta.total_seconds() - DEFAULT_CONFIRMATION_TTL_SECONDS) < 5


def test_store_approve_changes_status() -> None:
    store = ConfirmationStore()
    req = store.create(_make_request())
    approved = store.approve(req.confirmation_id, decided_by="user")
    assert approved is not None
    assert approved.status is ConfirmationStatus.APPROVED
    assert approved.is_approved
    assert approved.decided_by == "user"


def test_store_reject_changes_status() -> None:
    store = ConfirmationStore()
    req = store.create(_make_request())
    rejected = store.reject(req.confirmation_id)
    assert rejected is not None
    assert rejected.status is ConfirmationStatus.REJECTED


def test_store_approve_nonexistent_returns_none() -> None:
    store = ConfirmationStore()
    assert store.approve("nonexistent") is None


def test_store_double_approve_returns_none() -> None:
    store = ConfirmationStore()
    req = store.create(_make_request())
    store.approve(req.confirmation_id)
    second = store.approve(req.confirmation_id)
    assert second is None


def test_store_approve_after_reject_returns_none() -> None:
    store = ConfirmationStore()
    req = store.create(_make_request())
    store.reject(req.confirmation_id)
    assert store.approve(req.confirmation_id) is None


def test_store_list_pending_filters() -> None:
    store = ConfirmationStore()
    r1 = store.create(_make_request(tool_call_id="c1"))
    r2 = store.create(_make_request(tool_call_id="c2"))
    store.approve(r1.confirmation_id)
    pending = store.list_pending()
    assert len(pending) == 1
    assert pending[0].confirmation_id == r2.confirmation_id


def test_store_expire_marks_expired() -> None:
    store = ConfirmationStore()
    req = store.create(_make_request())
    expired = store.expire(req.confirmation_id)
    assert expired is not None
    assert expired.status is ConfirmationStatus.EXPIRED
    assert store.get(req.confirmation_id).status is ConfirmationStatus.EXPIRED


def test_store_auto_expires_on_get() -> None:
    store = ConfirmationStore()
    # Create with an already-expired timestamp.
    past = utc_now() - timedelta(seconds=100)
    req = _make_request()
    req = req.model_copy(update={"expires_at": past})
    store.create(req)
    fetched = store.get(req.confirmation_id)
    assert fetched.status is ConfirmationStatus.EXPIRED


def test_store_get_nonexistent_returns_none() -> None:
    store = ConfirmationStore()
    assert store.get("nope") is None


# ---------------------------------------------------------------------------
# LoopController pause/resume via MathAgent demos
# ---------------------------------------------------------------------------


def _make_math_controller(
    *, policy: DefaultToolPolicy, confirmation_store: ConfirmationStore, event_bus: InMemoryEventBus
) -> tuple[LoopController, AgentRegistry, ToolRegistry, ToolExecutor]:
    agent_registry = AgentRegistry()
    tool_registry = ToolRegistry()
    tool_executor = ToolExecutor(
        registry=tool_registry,
        policy=policy,
        confirmation_store=confirmation_store,
        event_bus=event_bus,
    )
    return (
        LoopController(
            discover_stage=DefaultDiscoverStage(),
            plan_stage=DefaultPlanStage(),
            execute_stage=DefaultExecuteStage(tool_executor=tool_executor),
            verify_stage=DefaultVerifyStage(verifier=MathVerifier()),
            iterate_stage=DefaultIterateStage(),
        ),
        agent_registry,
        tool_registry,
        tool_executor,
    )


async def _run_math_with_confirmation(*, event_bus: InMemoryEventBus | None = None) -> tuple:
    """Run a math loop that pauses for confirmation; return (controller, context, paused_result, store, bus)."""
    bus = event_bus or InMemoryEventBus()
    store = ConfirmationStore()
    policy = DefaultToolPolicy(require_confirmation_for_medium=True)
    controller, agent_registry, tool_registry, executor = _make_math_controller(
        policy=policy, confirmation_store=store, event_bus=bus
    )
    await tool_registry.register(CalculatorTool())
    # Make calculator MEDIUM risk so confirmation triggers.
    calc = tool_registry.get("calculator")
    calc._info = calc.info.model_copy(update={"risk_level": RiskLevel.MEDIUM})
    await agent_registry.register(MathAgent(tool_executor=executor))

    task = Task(input="What is 238 * 47?", agent_id=MATH_AGENT_ID, metadata={})
    state = LoopState(
        task_id=task.task_id,
        goal="What is 238 * 47?",
        success_criteria=["math_verified"],
        max_iterations=5,
    )
    context = LoopContext(
        state=state,
        task=task,
        agent_registry=agent_registry,
        tool_registry=tool_registry,
        event_bus=bus,
        policy=LoopPolicy(max_iterations=5, per_execution_timeout_seconds=10),
    )
    result = await controller.run(context)
    return controller, context, result, store, bus


async def test_loop_pauses_on_confirmation() -> None:
    _, _, result, store, _ = await _run_math_with_confirmation()
    assert result.final_status is LoopFinalStatus.WAITING_FOR_CONFIRMATION
    assert result.success is False
    pending = store.list_pending()
    assert len(pending) == 1
    assert pending[0].tool_name == "calculator"


async def test_loop_resume_after_approval_succeeds() -> None:
    controller, context, paused, store, _ = await _run_math_with_confirmation()
    assert paused.final_status is LoopFinalStatus.WAITING_FOR_CONFIRMATION
    pending = store.list_pending()
    assert len(pending) == 1
    confirm_id = pending[0].confirmation_id
    store.approve(confirm_id)
    resumed = await controller.resume_after_confirmation(context, confirm_id)
    assert resumed.final_status is LoopFinalStatus.SUCCESS
    assert resumed.final_response is not None
    assert resumed.final_response.get("result") == 11186


async def test_loop_resume_after_rejection_does_not_execute_tool() -> None:
    controller, context, paused, store, _ = await _run_math_with_confirmation()
    assert paused.final_status is LoopFinalStatus.WAITING_FOR_CONFIRMATION
    pending = store.list_pending()
    confirm_id = pending[0].confirmation_id
    store.reject(confirm_id)
    resumed = await controller.resume_after_rejection(context, confirm_id, reason="user said no")
    # After rejection, the tool is NOT executed. The loop iterates and hits
    # confirmation again (calculator is still MEDIUM), eventually pausing
    # or failing — but never producing a fabricated result.
    assert resumed.final_status is not LoopFinalStatus.SUCCESS
    assert resumed.success is False


async def test_confirmation_required_event_published() -> None:
    bus = InMemoryEventBus()
    events: list[Event] = []

    async def collect(event: Event) -> None:
        events.append(event)

    bus.subscribe(EventType.TOOL_CONFIRMATION_REQUIRED, collect)
    await _run_math_with_confirmation(event_bus=bus)
    required = [e for e in events if e.event_type is EventType.TOOL_CONFIRMATION_REQUIRED]
    assert len(required) >= 1
    assert required[0].payload.get("tool") == "calculator"


async def test_loop_waiting_for_confirmation_event_published() -> None:
    bus = InMemoryEventBus()
    events: list[Event] = []

    async def collect(event: Event) -> None:
        events.append(event)

    bus.subscribe(EventType.LOOP_WAITING_FOR_CONFIRMATION, collect)
    await _run_math_with_confirmation(event_bus=bus)
    waiting = [e for e in events if e.event_type is EventType.LOOP_WAITING_FOR_CONFIRMATION]
    assert len(waiting) >= 1


async def test_resumed_event_published_on_resume() -> None:
    bus = InMemoryEventBus()
    events: list[Event] = []

    async def collect(event: Event) -> None:
        events.append(event)

    bus.subscribe(EventType.LOOP_RESUMED, collect)
    controller, context, paused, store, _ = await _run_math_with_confirmation(event_bus=bus)
    pending = store.list_pending()
    store.approve(pending[0].confirmation_id)
    await controller.resume_after_confirmation(context, pending[0].confirmation_id)
    resumed = [e for e in events if e.event_type is EventType.LOOP_RESUMED]
    assert len(resumed) >= 1


# ---------------------------------------------------------------------------
# LoopService integration tests
# ---------------------------------------------------------------------------


def _make_loop_service(
    *, policy: DefaultToolPolicy | None = None, event_bus: InMemoryEventBus | None = None
) -> tuple[LoopService, AgentRegistry, ToolRegistry, InMemoryEventBus]:
    bus = event_bus or InMemoryEventBus()
    agent_registry = AgentRegistry()
    tool_registry = ToolRegistry()
    service = LoopService(
        task_manager=TaskManager(),
        agent_registry=agent_registry,
        tool_registry=tool_registry,
        event_bus=bus,
        session_manager=SessionManager(),
        tool_policy=policy or DefaultToolPolicy(require_confirmation_for_medium=True),
    )
    return service, agent_registry, tool_registry, bus


async def _setup_service_with_math(
    *, policy: DefaultToolPolicy | None = None
) -> tuple[LoopService, InMemoryEventBus]:
    service, agent_registry, tool_registry, bus = _make_loop_service(policy=policy)
    await tool_registry.register(CalculatorTool())
    calc = tool_registry.get("calculator")
    calc._info = calc.info.model_copy(update={"risk_level": RiskLevel.MEDIUM})
    await agent_registry.register(MathAgent(tool_executor=service._tool_executor))
    return service, bus


async def test_service_list_pending_confirmations_empty() -> None:
    service, _ = await _setup_service_with_math()
    assert service.list_pending_confirmations() == []


async def test_service_run_task_pauses_and_lists_pending() -> None:
    service, _ = await _setup_service_with_math()
    task = Task(input="What is 238 * 47?", agent_id=MATH_AGENT_ID, metadata={})
    result = await service.run_task_loop(
        task,
        goal="What is 238 * 47?",
        success_criteria=["math_verified"],
        verifier=MathVerifier(),
    )
    assert result.final_status is LoopFinalStatus.WAITING_FOR_CONFIRMATION
    pending = service.list_pending_confirmations()
    assert len(pending) == 1
    assert pending[0]["tool_name"] == "calculator"


async def test_service_approve_and_resume() -> None:
    service, _ = await _setup_service_with_math()
    task = Task(input="What is 238 * 47?", agent_id=MATH_AGENT_ID, metadata={})
    result = await service.run_task_loop(
        task,
        goal="What is 238 * 47?",
        success_criteria=["math_verified"],
        verifier=MathVerifier(),
    )
    assert result.final_status is LoopFinalStatus.WAITING_FOR_CONFIRMATION
    pending = service.list_pending_confirmations()
    confirm_id = pending[0]["confirmation_id"]

    approved = await service.approve_confirmation(confirm_id, decided_by="user")
    assert approved is True

    resumed = await service.resume_loop(task.task_id)
    assert resumed.final_status is LoopFinalStatus.SUCCESS
    assert resumed.final_response.get("result") == 11186


async def test_service_reject_and_resume() -> None:
    service, _ = await _setup_service_with_math()
    task = Task(input="What is 238 * 47?", agent_id=MATH_AGENT_ID, metadata={})
    result = await service.run_task_loop(
        task,
        goal="What is 238 * 47?",
        success_criteria=["math_verified"],
        verifier=MathVerifier(),
    )
    assert result.final_status is LoopFinalStatus.WAITING_FOR_CONFIRMATION
    pending = service.list_pending_confirmations()
    confirm_id = pending[0]["confirmation_id"]

    rejected = await service.reject_confirmation(confirm_id, decided_by="user")
    assert rejected is True

    resumed = await service.resume_loop_after_rejection(task.task_id, reason="no")
    assert resumed.final_status is not LoopFinalStatus.SUCCESS
    assert resumed.success is False


async def test_service_resume_nonexistent_task_raises() -> None:
    service, _ = await _setup_service_with_math()
    with pytest.raises(ValueError, match="No loop found"):
        await service.resume_loop("nonexistent-task")


async def test_service_resume_task_not_waiting_raises() -> None:
    service, agent_registry, tool_registry, _ = _make_loop_service(
        policy=DefaultToolPolicy(require_confirmation_for_medium=False)
    )
    await tool_registry.register(CalculatorTool())
    await agent_registry.register(MathAgent(tool_executor=service._tool_executor))
    task = Task(input="What is 2 + 2?", agent_id=MATH_AGENT_ID, metadata={})
    result = await service.run_task_loop(
        task,
        goal="What is 2 + 2?",
        success_criteria=["math_verified"],
        verifier=MathVerifier(),
    )
    assert result.final_status is LoopFinalStatus.SUCCESS
    with pytest.raises(ValueError, match="not waiting for confirmation"):
        await service.resume_loop(task.task_id)


async def test_service_approve_nonexistent_returns_false() -> None:
    service, _ = await _setup_service_with_math()
    assert await service.approve_confirmation("nonexistent") is False
    assert await service.reject_confirmation("nonexistent") is False


async def test_service_confirmation_approved_event_published() -> None:
    bus = InMemoryEventBus()
    events: list[Event] = []

    async def collect(event: Event) -> None:
        events.append(event)

    bus.subscribe(EventType.TOOL_CONFIRMATION_APPROVED, collect)
    service, _, _, _ = _make_loop_service(event_bus=bus)
    await service.tool_registry.register(CalculatorTool())
    calc = service.tool_registry.get("calculator")
    calc._info = calc.info.model_copy(update={"risk_level": RiskLevel.MEDIUM})
    await service.agent_registry.register(MathAgent(tool_executor=service._tool_executor))

    task = Task(input="What is 238 * 47?", agent_id=MATH_AGENT_ID, metadata={})
    await service.run_task_loop(
        task,
        goal="What is 238 * 47?",
        success_criteria=["math_verified"],
        verifier=MathVerifier(),
    )
    pending = service.list_pending_confirmations()
    confirm_id = pending[0]["confirmation_id"]
    await service.approve_confirmation(confirm_id, decided_by="user")
    approved_events = [e for e in events if e.event_type is EventType.TOOL_CONFIRMATION_APPROVED]
    assert len(approved_events) >= 1
    assert approved_events[0].payload.get("decided_by") == "user"
