"""Integration test: start the full application and verify /health."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app import __version__
from app.main import create_app


def test_app_starts_and_health_works() -> None:
    app = create_app()
    with TestClient(app) as client:
        # Lifespan startup has run: runtime should be started.
        assert app.state.runtime.is_started is True

        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["version"] == __version__

        # Runtime component is always reported as ok.
        components = {c["name"]: c["status"] for c in body["components"]}
        assert components["runtime"] == "ok"


def test_app_shutdown_runs() -> None:
    app = create_app()
    with TestClient(app) as client:
        assert app.state.runtime.is_started is True
        client.get("/health")
    # After context exit, lifespan shutdown has run.
    assert app.state.runtime.is_started is False
