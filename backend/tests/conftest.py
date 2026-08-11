"""Shared test fixtures and configuration."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator

# Ensure the backend package is importable as `app`.
BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(BACKEND_DIR))

# Force a testing environment before any app modules are imported.
os.environ.setdefault("APP_ENV", "testing")
os.environ.setdefault("APP_DEBUG", "false")
os.environ.setdefault("APP_LOG_LEVEL", "WARNING")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.config import get_settings  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> Iterator[None]:
    """Ensure settings reflect the test environment for each test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture()
def app():
    from app.main import create_app

    return create_app()


@pytest.fixture()
def client(app) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


# --- LLM test fixtures (shared across test_llm_* modules) ---


@pytest.fixture()
def event_bus():
    from app.events import InMemoryEventBus

    return InMemoryEventBus()


@pytest.fixture()
def captured_events(event_bus):
    """Return (event_bus, events) capturing LLM/model events only.

    No prompt contents are captured — only structured event payloads.
    """
    from app.events import EventType

    events: list = []
    captured_types = {
        EventType.LLM_REQUEST_STARTED,
        EventType.LLM_REQUEST_COMPLETED,
        EventType.LLM_REQUEST_FAILED,
        EventType.MODEL_SELECTED,
        EventType.MODEL_FALLBACK,
        EventType.MODEL_UNAVAILABLE,
    }

    async def handler(ev) -> None:
        if ev.event_type in captured_types:
            events.append(ev)

    event_bus.subscribe(None, handler)
    return event_bus, events
