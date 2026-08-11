"""Phase 5 tests: LLM status API and health endpoint LLM component.

Verifies the ``GET /api/v1/llm/status``, ``POST /api/v1/llm/health-check``
endpoints, and that ``GET /health`` reports the LLM component (degraded, not
error, when Ollama is unavailable — which it is in CI since no Ollama runs).
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_llm_status_endpoint_returns_snapshot(client: TestClient) -> None:
    response = client.get("/api/v1/llm/status")
    assert response.status_code == 200
    body = response.json()
    # In the test env, Ollama is unreachable -> degraded/disabled, but the
    # endpoint must still respond with structured data.
    assert "status" in body
    assert "provider" in body
    assert "model_available" in body
    assert "routing_policy" in body
    assert "capabilities" in body
    assert "installed_models" in body


def test_llm_status_no_secrets(client: TestClient) -> None:
    import json

    body = client.get("/api/v1/llm/status").json()
    blob = json.dumps(body)
    assert "api_key" not in blob.lower()
    assert "authorization" not in blob.lower()
    assert "token" not in blob.lower()


def test_llm_health_check_endpoint_returns_fresh_snapshot(client: TestClient) -> None:
    response = client.post("/api/v1/llm/health-check")
    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert "checked_at" in body


def test_health_includes_llm_component(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    names = {c["name"] for c in body["components"]}
    assert "llm" in names
    llm_component = next(c for c in body["components"] if c["name"] == "llm")
    # Ollama is unreachable in CI -> degraded or disabled, never "error".
    assert llm_component["status"] in ("ok", "degraded", "disabled")


def test_health_overall_ok_even_when_llm_unavailable(client: TestClient) -> None:
    """The overall /health status stays 'ok' when the LLM is unavailable."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_llm_status_endpoint_documented(client: TestClient) -> None:
    """The LLM status endpoint appears in the OpenAPI docs."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/api/v1/llm/status" in paths
    assert "/api/v1/llm/health-check" in paths
