"""Tests for the AgentRegistry."""

from __future__ import annotations

import pytest

from app.agents import AgentAlreadyRegisteredError, AgentNotFoundError
from app.agents.registry import AgentRegistry
from tests._fakes import EchoAgent


@pytest.fixture()
def registry() -> AgentRegistry:
    return AgentRegistry()


async def test_register_and_get(registry: AgentRegistry) -> None:
    agent = EchoAgent()
    await registry.register(agent)
    assert registry.exists("echo")
    assert registry.get("echo") is agent
    assert len(registry) == 1
    assert "echo" in registry


async def test_register_duplicate_raises(registry: AgentRegistry) -> None:
    await registry.register(EchoAgent())
    with pytest.raises(AgentAlreadyRegisteredError):
        await registry.register(EchoAgent())


async def test_unregister(registry: AgentRegistry) -> None:
    agent = EchoAgent()
    await registry.register(agent)
    removed = await registry.unregister("echo")
    assert removed is agent
    assert not registry.exists("echo")


async def test_unregister_missing_raises(registry: AgentRegistry) -> None:
    with pytest.raises(AgentNotFoundError):
        await registry.unregister("nope")


async def test_get_missing_raises(registry: AgentRegistry) -> None:
    with pytest.raises(AgentNotFoundError):
        registry.get("nope")


async def test_list_and_list_info(registry: AgentRegistry) -> None:
    await registry.register(EchoAgent("a", "A"))
    await registry.register(EchoAgent("b", "B"))
    assert {a.agent_id for a in registry.list()} == {"a", "b"}
    assert {i.name for i in registry.list_info()} == {"A", "B"}


async def test_find_by_capability(registry: AgentRegistry) -> None:
    from app.agents.interface import AgentCapability

    await registry.register(EchoAgent("a", "A", {AgentCapability.REASONING}))
    await registry.register(EchoAgent("b", "B", {AgentCapability.MATH}))
    found = registry.find_by_capability(AgentCapability.REASONING)
    assert [a.agent_id for a in found] == ["a"]


async def test_lifecycle_hooks_invoked(registry: AgentRegistry) -> None:
    calls: list[str] = []

    class Hooked(EchoAgent):
        async def on_register(self) -> None:
            calls.append("register")

        async def on_unregister(self) -> None:
            calls.append("unregister")

    agent = Hooked("hooked", "Hooked")
    await registry.register(agent)
    await registry.unregister("hooked")
    assert calls == ["register", "unregister"]
