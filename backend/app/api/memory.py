"""Memory REST API (Phase 6).

Exposes the persistent memory subsystem for inspection, manual retrieval,
and management. Endpoints:

* ``GET  /api/v1/memory/status``           — subsystem health snapshot
* ``GET  /api/v1/memory/memories``         — list memories (filtered)
* ``GET  /api/v1/memory/memories/{id}``    — fetch a single memory
* ``POST /api/v1/memory/memories``         — create a memory (via policy)
* ``DELETE /api/v1/memory/memories/{id}``  — delete a memory
* ``POST /api/v1/memory/retrieve``         — retrieve a bounded context
* ``POST /api/v1/memory/consolidate``      — run a consolidation pass
* ``GET  /api/v1/memory/conversations``    — list recent conversation turns

**Never** exposes secrets, API keys, or memory content for sensitive-pattern
matches (the write policy blocks those before storage). Content of stored
memories is returned for legitimate retrieval only.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.memory.types import MemorySource, MemoryType

logger = get_logger("api.memory")

router = APIRouter(prefix="/api/v1/memory", tags=["memory"])

__all__ = ["router"]


def _get_memory_manager(request: Request):
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None or not runtime.is_started:  # type: ignore[union-attr]
        raise HTTPException(status_code=503, detail="Runtime not initialized.")
    manager = getattr(runtime, "memory_manager", None)  # type: ignore[union-attr]
    if manager is None or not manager.is_enabled:
        raise HTTPException(status_code=503, detail="Memory subsystem disabled.")
    return runtime, manager


# --- Request/Response models ---


class MemoryCreateRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=10000)
    memory_type: str = Field(default="semantic")
    user_id: str | None = None
    session_id: str | None = None
    task_id: str | None = None
    agent_id: str | None = None
    source: str = Field(default="system")
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    user_id: str | None = None
    session_id: str | None = None
    task_id: str | None = None
    agent_id: str | None = None
    memory_types: list[str] = Field(default_factory=list)
    top_k: int | None = Field(default=None, ge=1, le=50)


# --- Endpoints ---


@router.get("/status")
async def memory_status(request: Request) -> dict:
    """Return the memory subsystem health snapshot."""
    runtime, _ = _get_memory_manager(request)
    health = runtime.memory_health  # type: ignore[union-attr]
    if health is None:
        return {"status": "disabled"}
    snap = await health.check()
    return snap.to_api_dict()


@router.get("/memories")
async def list_memories(
    request: Request,
    user_id: str | None = Query(default=None),
    session_id: str | None = Query(default=None),
    task_id: str | None = Query(default=None),
    memory_type: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
) -> dict:
    """List memories, optionally filtered by scope and type."""
    _, manager = _get_memory_manager(request)
    types: list[MemoryType] | None = None
    if memory_type:
        try:
            types = [MemoryType(memory_type)]
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Unknown memory_type: {memory_type}")
    records = await manager.list_by(
        user_id=user_id,
        session_id=session_id,
        task_id=task_id,
        memory_types=types,
        limit=limit,
    )
    return {
        "count": len(records),
        "memories": [r.model_dump(mode="json") for r in records],
    }


@router.get("/memories/{memory_id}")
async def get_memory(request: Request, memory_id: str) -> dict:
    """Fetch a single memory by ID."""
    _, manager = _get_memory_manager(request)
    record = await manager.get(memory_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Memory not found.")
    return record.model_dump(mode="json")


@router.post("/memories")
async def create_memory(request: Request, body: MemoryCreateRequest) -> dict:
    """Create a memory through the write policy (privacy + dedup)."""
    _, manager = _get_memory_manager(request)
    try:
        mem_type = MemoryType(body.memory_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown memory_type: {body.memory_type}")
    try:
        source = MemorySource(body.source)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown source: {body.source}")
    record = await manager.add(
        content=body.content,
        memory_type=mem_type,
        user_id=body.user_id,
        session_id=body.session_id,
        task_id=body.task_id,
        agent_id=body.agent_id,
        source=source,
        importance=body.importance,
        confidence=body.confidence,
        summary=body.summary,
        metadata=body.metadata,
    )
    if record is None:
        return {"status": "rejected", "reason": "policy_or_duplicate"}
    return record.model_dump(mode="json")


@router.delete("/memories/{memory_id}")
async def delete_memory(request: Request, memory_id: str) -> dict:
    """Delete a memory by ID."""
    _, manager = _get_memory_manager(request)
    deleted = await manager.delete(memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found.")
    return {"status": "deleted", "memory_id": memory_id}


@router.post("/retrieve")
async def retrieve(request: Request, body: RetrieveRequest) -> dict:
    """Retrieve a bounded memory context for a query."""
    _, manager = _get_memory_manager(request)
    types: list[MemoryType] | None = None
    if body.memory_types:
        try:
            types = [MemoryType(t) for t in body.memory_types]
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Unknown memory_type: {exc}")
    ctx = await manager.retrieve_context(
        body.query,
        user_id=body.user_id,
        session_id=body.session_id,
        task_id=body.task_id,
        agent_id=body.agent_id,
        memory_types=types,
    )
    return ctx.model_dump(mode="json")


@router.post("/consolidate")
async def consolidate(request: Request) -> dict:
    """Run a consolidation pass (expiry + promotion + dedup)."""
    _, manager = _get_memory_manager(request)
    report = await manager.consolidate()
    return {
        "deleted_expired": report.deleted_expired,
        "promoted": report.promoted,
        "merged_duplicates": report.merged_duplicates,
    }


@router.get("/conversations")
async def list_conversations(
    request: Request,
    session_id: str | None = Query(default=None),
    user_id: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
) -> dict:
    """List recent conversation messages."""
    _, manager = _get_memory_manager(request)
    messages = await manager.list_recent_conversation(
        session_id=session_id, user_id=user_id, limit=limit
    )
    summary = manager.summarize_conversation(messages)
    return {
        "count": len(messages),
        "summary": summary,
        "messages": [m.model_dump(mode="json") for m in messages],
    }
