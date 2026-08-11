"""Tests for the asynchronous event bus."""

from __future__ import annotations

import asyncio

import pytest

from app.events import Event, EventBus, EventType, InMemoryEventBus


@pytest.fixture()
def bus() -> EventBus:
    return InMemoryEventBus()


async def test_subscribe_and_publish_typed(bus: EventBus) -> None:
    received: list[Event] = []

    async def handler(event: Event) -> None:
        received.append(event)

    unsub = bus.subscribe(EventType.TASK_STARTED, handler)
    event = Event.create(EventType.TASK_STARTED, payload={"x": 1}, task_id="t1")
    await bus.publish(event)

    await asyncio.sleep(0)
    assert len(received) == 1
    assert received[0].event_type is EventType.TASK_STARTED
    assert received[0].payload == {"x": 1}
    assert received[0].task_id == "t1"
    assert received[0].event_id
    assert received[0].timestamp is not None
    # Verify unsubscribe works.
    unsub()
    await bus.publish(Event.create(EventType.TASK_STARTED))
    await asyncio.sleep(0)
    assert len(received) == 1


async def test_unsubscribe(bus: EventBus) -> None:
    received: list[Event] = []

    async def handler(event: Event) -> None:
        received.append(event)

    unsub = bus.subscribe(EventType.TASK_COMPLETED, handler)
    unsub()
    await bus.publish(Event.create(EventType.TASK_COMPLETED))
    await asyncio.sleep(0)
    assert received == []


async def test_multiple_subscribers(bus: EventBus) -> None:
    seen_a: list[Event] = []
    seen_b: list[Event] = []

    async def ha(e: Event) -> None:
        seen_a.append(e)

    async def hb(e: Event) -> None:
        seen_b.append(e)

    bus.subscribe(EventType.AGENT_STARTED, ha)
    bus.subscribe(EventType.AGENT_STARTED, hb)
    await bus.publish(Event.create(EventType.AGENT_STARTED))
    await asyncio.sleep(0.05)
    assert len(seen_a) == 1
    assert len(seen_b) == 1


async def test_wildcard_subscriber_receives_all(bus: EventBus) -> None:
    seen: list[Event] = []

    async def handler(e: Event) -> None:
        seen.append(e)

    bus.subscribe(None, handler)
    await bus.publish(Event.create(EventType.TASK_STARTED))
    await bus.publish(Event.create(EventType.TOOL_COMPLETED))
    await asyncio.sleep(0.05)
    assert len(seen) == 2


async def test_failing_handler_is_isolated(bus: EventBus) -> None:
    ok: list[Event] = []

    async def bad(e: Event) -> None:
        raise RuntimeError("boom")

    async def good(e: Event) -> None:
        ok.append(e)

    bus.subscribe(EventType.TASK_FAILED, bad)
    bus.subscribe(EventType.TASK_FAILED, good)
    await bus.publish(Event.create(EventType.TASK_FAILED))
    await asyncio.sleep(0.05)
    assert len(ok) == 1


async def test_publish_no_subscribers_is_noop(bus: EventBus) -> None:
    await bus.publish(Event.create(EventType.MEMORY_UPDATED))


async def test_sync_handler_supported(bus: EventBus) -> None:
    seen: list[Event] = []

    def handler(e: Event) -> None:
        seen.append(e)

    bus.subscribe(EventType.WORKFLOW_STARTED, handler)
    await bus.publish(Event.create(EventType.WORKFLOW_STARTED))
    await asyncio.sleep(0.05)
    assert len(seen) == 1


def test_all_required_event_types_exist() -> None:
    required = {
        "task.started",
        "task.completed",
        "task.failed",
        "agent.started",
        "agent.completed",
        "agent.failed",
        "tool.started",
        "tool.completed",
        "memory.updated",
        "workflow.started",
        "workflow.completed",
        "voice.status",
    }
    present = {e.value for e in EventType}
    assert required.issubset(present)


async def test_event_metadata_and_ids(bus: EventBus) -> None:
    seen: list[Event] = []

    async def handler(e: Event) -> None:
        seen.append(e)

    bus.subscribe(EventType.VOICE_STATUS, handler)
    e = Event.create(
        EventType.VOICE_STATUS,
        metadata={"source": "mic"},
        session_id="s1",
        agent_id="a1",
    )
    await bus.publish(e)
    await asyncio.sleep(0.05)
    assert seen[0].metadata == {"source": "mic"}
    assert seen[0].session_id == "s1"
    assert seen[0].agent_id == "a1"
    assert seen[0].event_id.startswith("evt_")
