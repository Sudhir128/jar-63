"""Health and version API routes."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app import __version__
from app.config import get_settings
from app.core.logging import get_logger
from app.schemas.common import ComponentStatus, HealthResponse, VersionResponse

logger = get_logger("api.health")

router = APIRouter(tags=["health"])

__all__ = ["router"]


@router.get("/health", response_model=HealthResponse)
async def health(app_request: Request) -> HealthResponse:
    """Return the application health status.

    The overall status is ``ok`` when the process is up. Component-level
    checks are best-effort and never cause the endpoint to error so that the
    endpoint itself remains a reliable liveness signal.
    """
    components: list[ComponentStatus] = []

    # Runtime readiness (always available in-process).
    components.append(ComponentStatus(name="runtime", status="ok"))

    # Optional dependency checks — degrade gracefully if unreachable.
    try:
        from app.database import check_async_db_connection

        db_ok = await check_async_db_connection()
        components.append(ComponentStatus(name="database", status="ok" if db_ok else "unreachable"))
    except Exception as exc:  # noqa: BLE001
        components.append(ComponentStatus(name="database", status="error", detail=str(exc)))

    try:
        from app.memory import check_redis_connection

        redis_ok = await check_redis_connection()
        components.append(ComponentStatus(name="redis", status="ok" if redis_ok else "unreachable"))
    except Exception as exc:  # noqa: BLE001
        components.append(ComponentStatus(name="redis", status="error", detail=str(exc)))

    # LLM (local Ollama). Degraded (not error) when unavailable — the runtime
    # continues with deterministic fallback.
    try:
        runtime = getattr(app_request.app.state, "runtime", None)  # type: ignore[arg-type]
        if runtime is not None and runtime.llm_health is not None:  # type: ignore[union-attr]
            snap = runtime.llm_health.snapshot  # type: ignore[union-attr]
            if snap.available:
                components.append(
                    ComponentStatus(
                        name="llm",
                        status="ok",
                        detail=f"{snap.provider}/{snap.model}",
                    )
                )
            else:
                components.append(
                    ComponentStatus(name="llm", status="degraded", detail=snap.detail)
                )
        else:
            components.append(ComponentStatus(name="llm", status="disabled"))
    except Exception as exc:  # noqa: BLE001
        components.append(ComponentStatus(name="llm", status="error", detail=str(exc)))

    # Memory subsystem (Phase 6). Degraded (not error) when PG unavailable.
    try:
        runtime = getattr(app_request.app.state, "runtime", None)  # type: ignore[arg-type]
        if runtime is not None and runtime.memory_health is not None:  # type: ignore[union-attr]
            mem_snap = runtime.memory_health.snapshot  # type: ignore[union-attr]
            if mem_snap is not None:
                components.append(
                    ComponentStatus(
                        name="memory",
                        status=mem_snap.status,
                        detail=f"records={mem_snap.record_count}",
                    )
                )
            else:
                components.append(ComponentStatus(name="memory", status="unknown"))
        else:
            components.append(ComponentStatus(name="memory", status="disabled"))
    except Exception as exc:  # noqa: BLE001
        components.append(ComponentStatus(name="memory", status="error", detail=str(exc)))

    return HealthResponse(status="ok", version=__version__, components=components)

@router.get("/version", response_model=VersionResponse)
async def version() -> VersionResponse:
    """Return application version metadata."""
    settings = get_settings()
    return VersionResponse(
        name=settings.app.name,
        version=__version__,
        api_version="v1",
        metadata={"env": settings.app.env.value},
    )
