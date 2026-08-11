"""Tests for the runtime manager composition and lifecycle."""

from __future__ import annotations

import pytest

from app.runtime import RuntimeManager


@pytest.fixture()
def runtime() -> RuntimeManager:
    return RuntimeManager()


async def test_start_and_shutdown(runtime: RuntimeManager) -> None:
    assert runtime.is_started is False
    await runtime.start()
    assert runtime.is_started is True
    assert runtime.dispatcher is not None
    assert runtime.workflow_manager is not None
    assert runtime.state.agent_registry_ready is True
    assert runtime.state.tool_registry_ready is True
    assert runtime.state.event_bus_ready is True
    await runtime.shutdown()
    assert runtime.is_started is False


async def test_start_is_idempotent(runtime: RuntimeManager) -> None:
    await runtime.start()
    dispatcher = runtime.dispatcher
    await runtime.start()
    assert runtime.dispatcher is dispatcher


async def test_shutdown_without_start_is_noop(runtime: RuntimeManager) -> None:
    await runtime.shutdown()
    assert runtime.is_started is False
