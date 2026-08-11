"""Tests for the task/loop API.

Exercises the REST surface of the Universal Loop Engine: creating tasks,
querying state/iterations, cancellation, and the loop list endpoint.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.agents.interface import (
    AgentCapability,
    AgentContext,
    AgentInfo,
    AgentInterface,
    AgentResult,
    AgentStatus,
)
from app.main import create_app


class ConstantApiAgent(AgentInterface):
    def __init__(self, agent_id: str, value: str) -> None:
        self._value = value
        self._info = AgentInfo(
            agent_id=agent_id,
            name=agent_id,
            description="Returns a constant value.",
            capabilities={AgentCapability.REASONING},
        )

    @property
    def info(self) -> AgentInfo:
        return self._info

    async def execute(self, context: AgentContext) -> AgentResult:
        return AgentResult(
            task_id=context.task_id,
            agent_id=self.agent_id,
            status=AgentStatus.COMPLETED,
            output=self._value,
        )


@pytest.fixture()
def app_with_agent():
    app = create_app()
    with TestClient(app) as client:
        runtime = app.state.runtime
        import asyncio

        asyncio.run(runtime.agent_registry.register(ConstantApiAgent("hello-agent", "hello")))
        yield client


def test_create_task_inline_success(app_with_agent: TestClient) -> None:
    response = app_with_agent.post(
        "/api/v1/tasks",
        json={
            "goal": "Return hello.",
            "agent_id": "hello-agent",
            "success_criteria": ["expected:hello"],
            "expected_output": "hello",
            "max_iterations": 5,
            "background": False,
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["status"] == "completed"
    assert body["result"] == "hello"
    assert body["loop_id"] is not None


def test_get_task_state_and_iterations(app_with_agent: TestClient) -> None:
    create = app_with_agent.post(
        "/api/v1/tasks",
        json={
            "goal": "Return hello.",
            "agent_id": "hello-agent",
            "expected_output": "hello",
            "max_iterations": 5,
        },
    )
    task_id = create.json()["task_id"]

    state = app_with_agent.get(f"/api/v1/tasks/{task_id}")
    assert state.status_code == 200
    sbody = state.json()
    assert sbody["loop_id"]
    assert sbody["goal"] == "Return hello."
    assert sbody["iteration_count"] >= 1

    iters = app_with_agent.get(f"/api/v1/tasks/{task_id}/iterations")
    assert iters.status_code == 200
    assert isinstance(iters.json(), list)


def test_get_task_result(app_with_agent: TestClient) -> None:
    create = app_with_agent.post(
        "/api/v1/tasks",
        json={
            "goal": "Return hello.",
            "agent_id": "hello-agent",
            "expected_output": "hello",
        },
    )
    task_id = create.json()["task_id"]
    result = app_with_agent.get(f"/api/v1/tasks/{task_id}/result")
    assert result.status_code == 200
    rbody = result.json()
    assert rbody["final_status"] == "success"
    assert rbody["success"] is True
    assert rbody["final_response"] == "hello"


def test_list_loops(app_with_agent: TestClient) -> None:
    app_with_agent.post(
        "/api/v1/tasks",
        json={"goal": "Return hello.", "agent_id": "hello-agent", "expected_output": "hello"},
    )
    response = app_with_agent.get("/api/v1/tasks")
    assert response.status_code == 200
    body = response.json()
    assert len(body["loops"]) >= 1
    assert "loop_id" in body["loops"][0]


def test_get_unknown_task_returns_404(app_with_agent: TestClient) -> None:
    response = app_with_agent.get("/api/v1/tasks/nonexistent")
    assert response.status_code == 404


def test_cancel_unknown_task_returns_404(app_with_agent: TestClient) -> None:
    response = app_with_agent.post("/api/v1/tasks/nonexistent/cancel")
    assert response.status_code == 404


def test_background_task_cancellable(app_with_agent: TestClient) -> None:
    """A background task can be cancelled via the API.

    We use a max-iterations loop that would otherwise run to completion; the
    cancel endpoint must return 200 and mark the loop as cancelled.
    """
    create = app_with_agent.post(
        "/api/v1/tasks",
        json={
            "goal": "Return 100.",
            "agent_id": "hello-agent",
            "expected_output": "100",  # never matches 'hello'
            "max_iterations": 50,
            "background": True,
        },
    )
    assert create.status_code == 201
    task_id = create.json()["task_id"]

    cancel = app_with_agent.post(f"/api/v1/tasks/{task_id}/cancel")
    assert cancel.status_code == 200
    cbody = cancel.json()
    assert cbody["cancelled"] is True
