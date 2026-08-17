"""Agent management REST API (Phase 7).

Exposes the agent orchestration subsystem:

* ``GET  /api/v1/agents``                  — list agent definitions
* ``GET  /api/v1/agents/{id}``             — get a definition
* ``POST /api/v1/agents``                  — register a definition
* ``PATCH /api/v1/agents/{id}``            — update a definition
* ``POST /api/v1/agents/{id}/lifecycle``   — set lifecycle state
* ``POST /api/v1/agents/{id}/version``     — create a new version
* ``GET  /api/v1/agents/{id}/executions``  — execution history (evidence)
* ``GET  /api/v1/agents/{id}/evaluations``  — evaluation history
* ``POST /api/v1/agents/{id}/evaluate``    — run an evaluation
* ``GET  /api/v1/agents/{id}/lifecycle/advice`` — lifecycle recommendation
* ``POST /api/v1/agents/run``              — orchestrate a run (router→
                                             dispatch→[evaluate])
* ``POST /api/v1/agents/dynamic``         — build a dynamic agent definition

The API never exposes secrets, instructions-as-secrets, or memory contents.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.agents.definitions import AgentDefinition, AgentKind, AgentLifecycleState
from app.agents.interface import AgentCapability
from app.agents.router import RoutingTask
from app.llm.models import PrivacyLevel
from app.runtime.models import Task, TaskStatus
from app.tools.interface import RiskLevel

logger = __import__("app.core.logging", fromlist=["get_logger"]).get_logger("api.agents")

router = APIRouter(prefix="/api/v1/agents", tags=["agents"])

__all__ = ["router"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_orchestrator(request: Request):
    runtime = request.app.state.runtime
    if runtime is None:
        raise HTTPException(status_code=503, detail="Runtime not initialized.")
    if getattr(runtime, "agent_orchestrator", None) is None:
        raise HTTPException(
            status_code=503, detail="Agent orchestration subsystem not initialized."
        )
    return runtime


def _definition_response(d: AgentDefinition) -> dict:
    return {
        "agent_id": d.agent_id,
        "name": d.name,
        "description": d.description,
        "purpose": d.purpose,
        "kind": d.kind.value,
        "capabilities": sorted(c.value for c in d.capabilities),
        "task_types": list(d.task_types),
        "required_tools": list(d.required_tools),
        "preferred_tools": list(d.preferred_tools),
        "privacy": d.privacy.value,
        "risk": d.risk.value,
        "lifecycle": d.lifecycle.value,
        "version": d.version,
        "dynamic": d.dynamic,
        "auto_activate": d.auto_activate,
        "usage": d.usage.model_dump(mode="json"),
        "created_at": d.created_at.isoformat() if d.created_at else None,
        "updated_at": d.updated_at.isoformat() if d.updated_at else None,
    }


def _execution_response(r) -> dict:
    return {
        "execution_id": r.execution_id,
        "agent_id": r.agent_id,
        "agent_version": r.agent_version,
        "task_id": r.task_id,
        "success": r.success,
        "final_status": r.final_status,
        "iterations_used": r.iterations_used,
        "tool_calls": r.tool_calls,
        "tool_failures": r.tool_failures,
        "policy_violations": r.policy_violations,
        "latency_ms": r.latency_ms,
        "cancelled": r.cancelled,
        "failure_reason": r.failure_reason,
        "created_at": r.created_at.isoformat() if r.created_at else None,
    }


def _evaluation_response(e) -> dict:
    return {
        "evaluation_id": e.evaluation_id,
        "agent_id": e.agent_id,
        "agent_version": e.agent_version,
        "overall_assessment": e.overall_assessment,
        "strengths": list(e.strengths),
        "weaknesses": list(e.weaknesses),
        "failure_reasons": list(e.failure_reasons),
        "recommended_changes": list(e.recommended_changes),
        "confidence": e.confidence,
        "recommendation": e.recommendation.value,
        "evaluator": e.evaluator,
        "created_at": e.created_at.isoformat() if e.created_at else None,
    }


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class CreateDefinitionRequest(BaseModel):
    agent_id: str
    name: str
    description: str = ""
    purpose: str = ""
    kind: AgentKind = AgentKind.GENERIC
    capabilities: list[AgentCapability] = Field(default_factory=list)
    task_types: list[str] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    preferred_tools: list[str] = Field(default_factory=list)
    instructions: str = ""
    constraints: list[str] = Field(default_factory=list)
    verification_strategy: str = ""
    model_requirements: list[str] = Field(default_factory=list)
    privacy: PrivacyLevel = PrivacyLevel.INTERNAL
    risk: RiskLevel = RiskLevel.LOW
    version: str = "0.1.0"
    dynamic: bool = False
    auto_activate: bool = False


class UpdateDefinitionRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    purpose: str | None = None
    instructions: str | None = None
    constraints: list[str] | None = None
    verification_strategy: str | None = None


class LifecycleRequest(BaseModel):
    lifecycle: AgentLifecycleState
    reason: str = ""


class VersionRequest(BaseModel):
    version: str
    reason: str = ""


class RunRequest(BaseModel):
    goal: str
    task_type: str | None = None
    requested_capabilities: list[AgentCapability] = Field(default_factory=list)
    agent_id: str | None = None
    max_iterations: int = 5
    expected_output: str | None = None
    evaluate: bool = False
    session_id: str | None = None


class DynamicBuildRequest(BaseModel):
    goal: str
    task_type: str | None = None
    requested_capabilities: list[AgentCapability] = Field(default_factory=list)
    privacy: PrivacyLevel = PrivacyLevel.INTERNAL


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("")
@router.get("/", include_in_schema=False)
async def list_agents(request: Request, lifecycle: str | None = None) -> dict:
    runtime = _get_orchestrator(request)
    state = AgentLifecycleState(lifecycle) if lifecycle else None
    items = await runtime.agent_catalog.list(lifecycle=state)
    return {"agents": [_definition_response(d) for d in items], "count": len(items)}


@router.get("/{agent_id}")
async def get_agent(request: Request, agent_id: str) -> dict:
    runtime = _get_orchestrator(request)
    try:
        d = await runtime.agent_catalog.get(agent_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _definition_response(d)


@router.post("", status_code=201)
@router.post("/", status_code=201, include_in_schema=False)
async def create_agent(request: Request, body: CreateDefinitionRequest) -> dict:
    runtime = _get_orchestrator(request)
    from app.agents.definitions import AgentDefinition

    try:
        d = AgentDefinition(
            agent_id=body.agent_id,
            name=body.name,
            description=body.description,
            purpose=body.purpose,
            kind=body.kind,
            capabilities=set(body.capabilities),
            task_types=list(body.task_types),
            required_tools=list(body.required_tools),
            preferred_tools=list(body.preferred_tools),
            instructions=body.instructions,
            constraints=list(body.constraints),
            verification_strategy=body.verification_strategy,
            model_requirements=list(body.model_requirements),
            privacy=body.privacy,
            risk=body.risk,
            version=body.version,
            dynamic=body.dynamic,
            auto_activate=body.auto_activate,
        )
        await runtime.agent_catalog.register(d)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _definition_response(d)


@router.patch("/{agent_id}")
async def update_agent(request: Request, agent_id: str, body: UpdateDefinitionRequest) -> dict:
    runtime = _get_orchestrator(request)
    fields = body.model_dump(exclude_none=True)
    try:
        d = await runtime.agent_catalog.update_metadata(agent_id, **fields)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _definition_response(d)


@router.post("/{agent_id}/lifecycle")
async def set_lifecycle(request: Request, agent_id: str, body: LifecycleRequest) -> dict:
    runtime = _get_orchestrator(request)
    try:
        d = await runtime.agent_catalog.set_lifecycle(agent_id, body.lifecycle, reason=body.reason)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _definition_response(d)


@router.post("/{agent_id}/version")
async def create_version(request: Request, agent_id: str, body: VersionRequest) -> dict:
    runtime = _get_orchestrator(request)
    try:
        d = await runtime.agent_catalog.create_version(
            agent_id, version=body.version, reason=body.reason
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _definition_response(d)


@router.get("/{agent_id}/executions")
async def list_executions(request: Request, agent_id: str, limit: int = 50) -> dict:
    runtime = _get_orchestrator(request)
    rows = await runtime.agent_execution_store.list_for_agent(agent_id, limit=limit)
    return {
        "executions": [_execution_response(r) for r in rows],
        "count": len(rows),
    }


@router.get("/{agent_id}/evaluations")
async def list_evaluations(request: Request, agent_id: str, limit: int = 50) -> dict:
    runtime = _get_orchestrator(request)
    rows = await runtime.agent_evaluation_store.list_for_agent(agent_id, limit=limit)
    return {
        "evaluations": [_evaluation_response(e) for e in rows],
        "count": len(rows),
    }


@router.post("/{agent_id}/evaluate")
async def evaluate_agent(request: Request, agent_id: str) -> dict:
    runtime = _get_orchestrator(request)
    try:
        d = await runtime.agent_catalog.get(agent_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    evaluation = await runtime.agent_evaluator.evaluate(d)
    return _evaluation_response(evaluation)


@router.get("/{agent_id}/lifecycle/advice")
async def lifecycle_advice(request: Request, agent_id: str) -> dict:
    runtime = _get_orchestrator(request)
    try:
        d = await runtime.agent_catalog.get(agent_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    advice = await runtime.lifecycle_manager.advise(d)
    return {
        "agent_id": advice.agent_id,
        "recommendation": advice.recommendation.value,
        "reason": advice.reason,
        "evidence": advice.evidence,
    }


@router.post("/run")
async def run_agent(request: Request, body: RunRequest) -> dict:
    runtime = _get_orchestrator(request)
    task = Task(
        task_id=__import__("app.core.identifiers", fromlist=["generate_id"]).generate_id("task"),
        agent_id=body.agent_id or "",
        input=body.goal,
        status=TaskStatus.PENDING,
        session_id=body.session_id,
    )
    routing = RoutingTask(
        goal=body.goal,
        task_type=body.task_type,
        requested_capabilities=set(body.requested_capabilities),
        agent_id=body.agent_id,
        privacy=PrivacyLevel.INTERNAL,
        metadata={"session_id": body.session_id} if body.session_id else {},
    )
    result = await runtime.agent_orchestrator.run(
        task,
        routing,
        evaluate=body.evaluate,
        max_iterations=body.max_iterations,
        expected_output=body.expected_output,
    )
    return {
        "routing": {
            "agent_id": result.routing.agent_id,
            "reason": result.routing.reason,
            "score": result.routing.score,
            "used_llm": result.routing.used_llm,
        },
        "dynamic_created": result.dynamic_created,
        "dispatch": {
            "agent_id": result.dispatch.definition.agent_id if result.dispatch else None,
            "success": result.dispatch.record.success if result.dispatch else False,
            "final_status": result.dispatch.record.final_status if result.dispatch else None,
            "iterations": result.dispatch.record.iterations_used if result.dispatch else 0,
            "tool_calls": result.dispatch.record.tool_calls if result.dispatch else 0,
        }
        if result.dispatch
        else None,
        "evaluation": (
            {
                "recommendation": result.evaluation.recommendation.value,
                "confidence": result.evaluation.confidence,
                "assessor": result.evaluation.evaluator,
            }
            if result.evaluation
            else None
        ),
    }


@router.post("/dynamic", status_code=201)
async def build_dynamic(request: Request, body: DynamicBuildRequest) -> dict:
    runtime = _get_orchestrator(request)
    if runtime.dynamic_agent_builder is None:
        raise HTTPException(status_code=403, detail="Dynamic agent building is disabled.")
    try:
        result = await runtime.dynamic_agent_builder.build_for(
            body.goal,
            task_type=body.task_type,
            requested_capabilities=set(body.requested_capabilities),
            privacy=body.privacy,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "definition": _definition_response(result.definition),
        "created": result.created,
        "used_llm": result.used_llm,
    }
