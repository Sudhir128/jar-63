"""LLM status API route (Phase 5).

Exposes the current LLM subsystem health, selected provider/model, model
availability, capabilities, routing policy, and fallback availability.

**Never** exposes API keys, authorization headers, raw prompts, or
completions. Only operational metadata.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.core.logging import get_logger

logger = get_logger("api.llm")

router = APIRouter(prefix="/api/v1/llm", tags=["llm"])

__all__ = ["router"]


def _get_runtime(request: Request):
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None or not runtime.is_started:  # type: ignore[union-attr]
        raise HTTPException(status_code=503, detail="Runtime not initialized.")
    return runtime


@router.get("/status")
async def llm_status(request: Request) -> dict:
    """Return the current LLM subsystem status.

    Returns provider, model, availability, capabilities, routing policy, and
    fallback availability. No secrets are ever included.
    """
    runtime = _get_runtime(request)
    health = runtime.llm_health  # type: ignore[union-attr]
    if health is None:
        return {
            "status": "disabled",
            "provider": None,
            "model": None,
            "model_available": False,
            "health_status": "unavailable",
            "installed_models": [],
            "capabilities": [],
            "routing_policy": "local_first",
            "fallback_available": True,
            "cloud_enabled": False,
            "detail": "LLM subsystem not initialized.",
        }
    snapshot = health.snapshot
    result = snapshot.to_api_dict()
    # Add the selected model from the router (if any) for observability.
    result["selected_provider"] = snapshot.provider if snapshot.available else None
    result["selected_model"] = snapshot.model if snapshot.available else None
    return result


@router.post("/health-check")
async def trigger_health_check(request: Request) -> dict:
    """Trigger an on-demand LLM health check and return the fresh snapshot."""
    runtime = _get_runtime(request)
    health = runtime.llm_health  # type: ignore[union-attr]
    if health is None:
        raise HTTPException(status_code=503, detail="LLM health checker not initialized.")
    snapshot = await health.check()
    return snapshot.to_api_dict()
