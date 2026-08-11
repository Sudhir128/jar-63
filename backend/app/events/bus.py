"""Event bus interfaces and an asynchronous in-memory implementation.

The interface (:class:`EventBus`) allows the implementation to be swapped
later (e.g. a Redis-backed or persistent bus) without changing consumers.
"""

from __future__ import annotations

import abc
from collections import defaultdict
from collections.abc import Awaitable, Callable

import anyio

from app.core.logging import get_logger
from app.events.types import Event, EventType

logger = get_logger("events")

__all__ = ["EventHandler", "EventBus", "InMemoryEventBus", "get_event_bus"]

# An event handler is any callable returning an awaitable when invoked with an event.
EventHandler = Callable[[Event], Awaitable[None]] | Callable[[Event], None]


class EventBus(abc.ABC):
    """Abstract event bus contract."""

    @abc.abstractmethod
    async def publish(self, event: Event) -> None:
        """Publish an event to all matching subscribers."""

    @abc.abstractmethod
    def subscribe(self, event_type: EventType | None, handler: EventHandler) -> Callable[[], None]:
        """Subscribe ``handler`` to events of ``event_type``.

        If ``event_type`` is ``None`` the handler receives all events.
        Returns an unsubscribe callable.
        """

    def unsubscribe(self, handler: EventHandler) -> None:
        """Remove ``handler`` from all subscriptions (default best-effort)."""
        raise NotImplementedError


class InMemoryEventBus(EventBus):
    """Asynchronous, in-process event bus.

    Handlers are invoked concurrently per-event via a task group. A handler
    raising an exception is logged and isolated; it never prevents other
    subscribers from receiving the event.
    """

    def __init__(self) -> None:
        self._handlers: dict[EventType | None, set[EventHandler]] = defaultdict(set)
        self._lock = anyio.Lock()

    def subscribe(self, event_type: EventType | None, handler: EventHandler) -> Callable[[], None]:
        self._handlers[event_type].add(handler)

        def _unsubscribe() -> None:
            self._handlers[event_type].discard(handler)

        return _unsubscribe

    def unsubscribe(self, handler: EventHandler) -> None:
        for handlers in list(self._handlers.values()):
            handlers.discard(handler)

    def _matching_handlers(self, event_type: EventType) -> list[EventHandler]:
        return list(self._handlers.get(event_type, set())) + list(self._handlers.get(None, set()))

    async def publish(self, event: Event) -> None:
        handlers = self._matching_handlers(event.event_type)
        if not handlers:
            return

        async def _run(h: EventHandler, ev: Event) -> None:
            try:
                result = h(ev)
                if hasattr(result, "__await__"):
                    await result  # type: ignore[func-returns-value]
            except Exception as exc:  # noqa: BLE001 - isolate failing handlers
                logger.bind(
                    event=ev.event_type.value,
                    event_id=ev.event_id,
                    error=type(exc).__name__,
                ).warning("Event handler failed: {}", str(exc))

        async with anyio.create_task_group() as tg:
            for h in handlers:
                tg.start_soon(_run, h, event)


_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Return the process-wide event bus (lazily created)."""
    global _bus
    if _bus is None:
        _bus = InMemoryEventBus()
    return _bus


def set_event_bus(bus: EventBus) -> None:
    """Override the process-wide bus (used in tests / future providers)."""
    global _bus
    _bus = bus
