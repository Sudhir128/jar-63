"""Tests for the FastAPI health and version endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app import __version__


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__
    assert isinstance(body["components"], list)
    # Runtime is always reported.
    names = {c["name"] for c in body["components"]}
    assert "runtime" in names


def test_version_endpoint(client: TestClient) -> None:
    response = client.get("/version")
    assert response.status_code == 200
    body = response.json()
    assert body["version"] == __version__
    assert body["api_version"] == "v1"


def test_openapi_docs_available(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200
    assert client.get("/redoc").status_code == 200
    assert client.get("/openapi.json").status_code == 200
