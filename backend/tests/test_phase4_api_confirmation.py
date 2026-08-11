"""Phase 4 tests: API confirmation resume endpoints.

Tests cover:
* GET /api/v1/tasks/confirmations/pending (empty and with pending).
* POST /api/v1/tasks/confirmations/{id}/approve.
* POST /api/v1/tasks/confirmations/{id}/reject.
* POST /api/v1/tasks/{task_id}/resume.
* 404 for nonexistent confirmations.
* Typed ConfirmationResponse / ResumeTaskResponse.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from app.agents.math import MATH_AGENT_ID, MathAgent
from app.main import create_app
from app.tools.interface import RiskLevel
from app.tools.policy import DefaultToolPolicy


@pytest.fixture()
def app_with_math_confirmation():
    """Create an app with a MathAgent and a confirmation-requiring policy."""
    app = create_app()
    with TestClient(app) as client:
        runtime = app.state.runtime

        # Replace the loop_service with one that requires confirmation
        # for MEDIUM-risk tools and uses the MathVerifier by default.
        from app.agents.math.verifier import MathVerifier
        from app.runtime.loop.service import LoopService

        runtime.loop_service = LoopService(
            task_manager=runtime.task_manager,
            agent_registry=runtime.agent_registry,
            tool_registry=runtime.tool_registry,
            event_bus=runtime.event_bus,
            session_manager=runtime.session_manager,
            tool_policy=DefaultToolPolicy(require_confirmation_for_medium=True),
        )
        # Override make_controller so all tasks use the MathVerifier.
        _orig_make_controller = runtime.loop_service.make_controller

        def _math_make_controller(*, verifier=None, stop_conditions=None):
            return _orig_make_controller(
                verifier=verifier or MathVerifier(),
                stop_conditions=stop_conditions,
            )

        runtime.loop_service.make_controller = _math_make_controller

        async def setup() -> None:
            # The calculator is already registered by bootstrap; just set
            # its risk level to MEDIUM so confirmation is triggered.
            calc = runtime.tool_registry.get("calculator")
            calc._info = calc.info.model_copy(update={"risk_level": RiskLevel.MEDIUM})
            # Register the real MathAgent (the bootstrap's demo.math is a
            # scripted agent, not the real one).
            import contextlib

            with contextlib.suppress(Exception):
                await runtime.agent_registry.register(
                    MathAgent(tool_executor=runtime.loop_service._tool_executor)
                )

        asyncio.run(setup())
        yield client


def test_list_pending_confirmations_empty(app_with_math_confirmation: TestClient) -> None:
    response = app_with_math_confirmation.get("/api/v1/tasks/confirmations/pending")
    assert response.status_code == 200
    assert response.json() == []


def test_approve_nonexistent_confirmation_returns_404(
    app_with_math_confirmation: TestClient,
) -> None:
    response = app_with_math_confirmation.post("/api/v1/tasks/confirmations/nonexistent-id/approve")
    assert response.status_code == 404


def test_reject_nonexistent_confirmation_returns_404(
    app_with_math_confirmation: TestClient,
) -> None:
    response = app_with_math_confirmation.post("/api/v1/tasks/confirmations/nonexistent-id/reject")
    assert response.status_code == 404


def test_full_confirmation_lifecycle_via_api(
    app_with_math_confirmation: TestClient,
) -> None:
    """Create a task that pauses for confirmation, then approve via API."""
    # 1. Create the task (inline — it will pause).
    create = app_with_math_confirmation.post(
        "/api/v1/tasks",
        json={
            "goal": "What is 238 * 47?",
            "agent_id": MATH_AGENT_ID,
            "success_criteria": ["math_verified"],
            "max_iterations": 5,
            "background": False,
        },
    )
    assert create.status_code == 201, create.text
    body = create.json()
    # The task should be waiting for confirmation.
    assert body["status"] == "waiting_for_confirmation"
    task_id = body["task_id"]

    # 2. List pending confirmations.
    pending = app_with_math_confirmation.get("/api/v1/tasks/confirmations/pending")
    assert pending.status_code == 200
    pending_list = pending.json()
    assert len(pending_list) >= 1
    confirm = pending_list[0]
    assert confirm["tool"] == "calculator"
    assert confirm["status"] == "pending"
    assert confirm["confirmation_id"]
    confirm_id = confirm["confirmation_id"]

    # 3. Approve the confirmation via API.
    approve = app_with_math_confirmation.post(f"/api/v1/tasks/confirmations/{confirm_id}/approve")
    assert approve.status_code == 200, approve.text
    approved_body = approve.json()
    assert approved_body["status"] == "approved"
    assert approved_body["tool"] == "calculator"

    # 4. Verify the task result is now successful.
    result = app_with_math_confirmation.get(f"/api/v1/tasks/{task_id}/result")
    assert result.status_code == 200
    result_body = result.json()
    assert result_body["final_status"] == "success"
    assert result_body["final_response"]["result"] == 11186


def test_reject_confirmation_via_api(
    app_with_math_confirmation: TestClient,
) -> None:
    """Create a task that pauses for confirmation, then reject via API."""
    create = app_with_math_confirmation.post(
        "/api/v1/tasks",
        json={
            "goal": "What is 238 * 47?",
            "agent_id": MATH_AGENT_ID,
            "success_criteria": ["math_verified"],
            "max_iterations": 3,
            "background": False,
        },
    )
    assert create.status_code == 201
    body = create.json()
    assert body["status"] == "waiting_for_confirmation"

    pending = app_with_math_confirmation.get("/api/v1/tasks/confirmations/pending")
    pending_list = pending.json()
    assert len(pending_list) >= 1
    confirm_id = pending_list[0]["confirmation_id"]

    reject = app_with_math_confirmation.post(f"/api/v1/tasks/confirmations/{confirm_id}/reject")
    assert reject.status_code == 200, reject.text
    rejected_body = reject.json()
    assert rejected_body["status"] == "rejected"

    # The task should NOT be successful.
    task_id = body["task_id"]
    result = app_with_math_confirmation.get(f"/api/v1/tasks/{task_id}/result")
    result_body = result.json()
    assert result_body["final_status"] != "success"
    assert result_body["success"] is False


def test_resume_endpoint_for_nonexistent_task_returns_error(
    app_with_math_confirmation: TestClient,
) -> None:
    """Resuming a nonexistent task returns a conflict/client error."""
    response = app_with_math_confirmation.post("/api/v1/tasks/nonexistent/resume")
    assert response.status_code in (404, 409)


def test_confirmation_response_is_typed(app_with_math_confirmation: TestClient) -> None:
    """Verify ConfirmationResponse has all required fields."""
    create = app_with_math_confirmation.post(
        "/api/v1/tasks",
        json={
            "goal": "What is 5 * 5?",
            "agent_id": MATH_AGENT_ID,
            "success_criteria": ["math_verified"],
            "max_iterations": 3,
            "background": False,
        },
    )
    assert create.status_code == 201
    pending = app_with_math_confirmation.get("/api/v1/tasks/confirmations/pending")
    confirm = pending.json()[0]
    # Verify all required fields are present.
    assert "confirmation_id" in confirm
    assert "task_id" in confirm
    assert "tool" in confirm
    assert "risk" in confirm
    assert "status" in confirm
    assert "reason" in confirm
