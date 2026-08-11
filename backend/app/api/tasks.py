"""Task and loop API routes.

Exposes the Universal Loop Engine through a REST surface. Tasks may be run
inline (synchronous result) or in the background (status/iteration polling +
cancellation).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.core.logging import get_logger
from app.runtime.loop.verification import ExactMatchVerifier
from app.runtime.models import Task
from app.schemas.tasks import (
    CancelTaskResponse,
    ConfirmationResponse,
    CreateTaskRequest,
    LoopIterationResponse,
    LoopListResponse,
    LoopResultResponse,
    LoopStateResponse,
    ResumeTaskResponse,
    TaskResponse,
)

logger = get_logger("api.tasks")

router = APIRouter(prefix="/api/v1/tasks", tags=["tasks"])

__all__ = ["router"]


def _get_loop_service(request: Request):
    runtime = request.app.state.runtime
    if runtime is None or runtime.loop_service is None:  # type: ignore[attr-defined]
        raise HTTPException(status_code=503, detail="Loop service not initialized.")
    return runtime.loop_service  # type: ignore[attr-defined]


def _iteration_response(record) -> LoopIterationResponse:
    return LoopIterationResponse(
        iteration_number=record.iteration_number,
        stage=record.stage,
        action=record.action.model_dump() if record.action else None,
        result=record.result.model_dump() if record.result else None,
        verification=record.verification.model_dump() if record.verification else None,
        changes=list(record.changes or []),
        errors=list(record.errors or []),
    )


def _state_response(handle, service) -> LoopStateResponse:
    state = handle.controller.state
    iterations = [_iteration_response(r) for r in state.iteration_history]
    final_result = handle.result.model_dump() if handle.result else None
    return LoopStateResponse(
        loop_id=handle.loop_id,
        task_id=handle.task_id,
        session_id=state.session_id,
        goal=state.goal,
        status=state.status.value,
        current_stage=state.current_stage.value,
        iteration_count=state.iteration_count,
        max_iterations=state.max_iterations,
        next_action=state.next_action.model_dump() if state.next_action else None,
        completed_steps=list(state.completed_steps),
        failed_steps=list(state.failed_steps),
        changes=list(state.changes),
        remaining_work=list(state.remaining_work),
        blockers=list(state.blockers),
        last_error=state.last_error,
        iterations=iterations,
        final_result=final_result,
    )


@router.post("", response_model=TaskResponse, status_code=201)
@router.post("/", response_model=TaskResponse, status_code=201, include_in_schema=False)
async def create_task(request: Request, body: CreateTaskRequest) -> TaskResponse:
    """Create a task and run it through the Universal Loop Engine."""
    service = _get_loop_service(request)
    task = Task(
        input=body.input if body.input is not None else body.goal,
        agent_id=body.agent_id,
        session_id=body.session_id,
    )
    verifier = (
        ExactMatchVerifier(check_name="expected_output")
        if body.expected_output is not None
        else None
    )
    if body.background:
        handle = await service.start_task_loop(
            task,
            goal=body.goal,
            success_criteria=body.success_criteria,
            max_iterations=body.max_iterations,
            expected_output=body.expected_output,
            verifier=verifier,
        )
        return TaskResponse(
            task_id=task.task_id,
            session_id=task.session_id,
            agent_id=task.agent_id,
            status=task.status.value,
            loop_id=handle.loop_id,
        )
    result = await service.run_task_loop(
        task,
        goal=body.goal,
        success_criteria=body.success_criteria,
        max_iterations=body.max_iterations,
        expected_output=body.expected_output,
        verifier=verifier,
    )
    return TaskResponse(
        task_id=task.task_id,
        session_id=task.session_id,
        agent_id=task.agent_id,
        status=task.status.value,
        loop_id=service.get_handle(task.task_id).loop_id
        if service.get_handle(task.task_id)
        else None,
        result=result.final_response,
        error=result.failure_reason,
    )


@router.get("", response_model=LoopListResponse)
@router.get("/", response_model=LoopListResponse, include_in_schema=False)
async def list_loops(request: Request) -> LoopListResponse:
    """List all loops known to the loop service."""
    service = _get_loop_service(request)
    return LoopListResponse(loops=service.list_loops())


@router.get("/{task_id}", response_model=LoopStateResponse)
async def get_task_state(request: Request, task_id: str) -> LoopStateResponse:
    """Return the current loop state for a task, including iteration history."""
    service = _get_loop_service(request)
    handle = service.get_handle(task_id)
    if handle is None:
        raise HTTPException(status_code=404, detail=f"No loop found for task '{task_id}'.")
    return _state_response(handle, service)


@router.get("/{task_id}/result", response_model=LoopResultResponse)
async def get_task_result(request: Request, task_id: str) -> LoopResultResponse:
    """Return the final loop result for a task, if available."""
    service = _get_loop_service(request)
    handle = service.get_handle(task_id)
    if handle is None:
        raise HTTPException(status_code=404, detail=f"No loop found for task '{task_id}'.")
    if handle.result is None:
        raise HTTPException(status_code=409, detail="Loop has not completed yet.")
    r = handle.result
    return LoopResultResponse(
        loop_id=r.loop_id,
        task_id=r.task_id,
        final_status=r.final_status.value,
        success=r.success,
        iterations_used=r.iterations_used,
        final_response=r.final_response,
        verification_evidence=[e.model_dump() for e in r.verification_evidence],
        completed_work=list(r.completed_work),
        remaining_work=list(r.remaining_work),
        failure_reason=r.failure_reason,
        stopped_reason=r.stopped_reason,
    )


@router.get("/{task_id}/iterations", response_model=list[LoopIterationResponse])
async def get_task_iterations(request: Request, task_id: str) -> list[LoopIterationResponse]:
    """Return the iteration history for a task."""
    service = _get_loop_service(request)
    handle = service.get_handle(task_id)
    if handle is None:
        raise HTTPException(status_code=404, detail=f"No loop found for task '{task_id}'.")
    return [_iteration_response(r) for r in handle.controller.state.iteration_history]


@router.post("/{task_id}/cancel", response_model=CancelTaskResponse)
async def cancel_task(request: Request, task_id: str) -> CancelTaskResponse:
    """Request cancellation of a running loop."""
    service = _get_loop_service(request)
    cancelled = service.cancel(task_id)
    if not cancelled:
        raise HTTPException(status_code=404, detail=f"No loop found for task '{task_id}'.")
    return CancelTaskResponse(
        task_id=task_id,
        cancelled=True,
        message="Cancellation requested; the loop will stop at the next stage boundary.",
    )


def _confirmation_response(data: dict) -> ConfirmationResponse:
    """Build a typed, safe :class:`ConfirmationResponse` from a raw dict."""
    return ConfirmationResponse(
        confirmation_id=data.get("confirmation_id", ""),
        task_id=data.get("task_id"),
        loop_id=data.get("loop_id"),
        tool=data.get("tool_name", "unknown"),
        risk=data.get("risk_level", "low"),
        reason=data.get("reason", ""),
        status=data.get("status", "pending"),
        iteration=data.get("iteration"),
    )


def _resume_response(task_id: str, result) -> ResumeTaskResponse:
    return ResumeTaskResponse(
        task_id=task_id,
        loop_id=result.loop_id,
        final_status=result.final_status.value,
        success=result.success,
        iterations_used=result.iterations_used,
        final_response=result.final_response,
        failure_reason=result.failure_reason,
        stopped_reason=result.stopped_reason,
    )


@router.get("/confirmations/pending", response_model=list[ConfirmationResponse])
async def list_pending_confirmations(request: Request) -> list[ConfirmationResponse]:
    """List all pending tool confirmation requests."""
    service = _get_loop_service(request)
    return [_confirmation_response(r) for r in service.list_pending_confirmations()]


@router.post("/confirmations/{confirmation_id}/approve", response_model=ConfirmationResponse)
async def approve_confirmation(
    request: Request, confirmation_id: str
) -> ConfirmationResponse:
    """Approve a pending tool confirmation and resume the paused loop."""
    service = _get_loop_service(request)
    approved = await service.approve_confirmation(confirmation_id)
    if not approved:
        raise HTTPException(
            status_code=404,
            detail=f"Confirmation '{confirmation_id}' not found or not pending.",
        )
    # Resume the loop for the task associated with this confirmation.
    pending = service._confirmation_store.get(confirmation_id)  # noqa: SLF001
    task_id = pending.task_id if pending else None
    if task_id and service.get_handle(task_id) is not None:
        try:
            await service.resume_loop(task_id)
        except ValueError:
            pass  # Loop was not in a pausable state; approval still recorded.
    req = service._confirmation_store.get(confirmation_id)  # noqa: SLF001
    data = req.model_dump() if req else {"confirmation_id": confirmation_id, "status": "approved"}
    return _confirmation_response(data)


@router.post("/confirmations/{confirmation_id}/reject", response_model=ConfirmationResponse)
async def reject_confirmation(
    request: Request, confirmation_id: str
) -> ConfirmationResponse:
    """Reject a pending tool confirmation and resume the loop (tool not executed)."""
    service = _get_loop_service(request)
    rejected = await service.reject_confirmation(confirmation_id)
    if not rejected:
        raise HTTPException(
            status_code=404,
            detail=f"Confirmation '{confirmation_id}' not found or not pending.",
        )
    pending = service._confirmation_store.get(confirmation_id)  # noqa: SLF001
    task_id = pending.task_id if pending else None
    if task_id and service.get_handle(task_id) is not None:
        try:
            await service.resume_loop_after_rejection(task_id, reason="rejected by user")
        except ValueError:
            pass
    req = service._confirmation_store.get(confirmation_id)  # noqa: SLF001
    data = req.model_dump() if req else {"confirmation_id": confirmation_id, "status": "rejected"}
    return _confirmation_response(data)


@router.post("/{task_id}/resume", response_model=ResumeTaskResponse)
async def resume_task(request: Request, task_id: str) -> ResumeTaskResponse:
    """Resume a paused loop after its confirmation was approved."""
    service = _get_loop_service(request)
    try:
        result = await service.resume_loop(task_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _resume_response(task_id, result)
