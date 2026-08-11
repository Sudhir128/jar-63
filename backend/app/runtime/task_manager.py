"""Task manager: creates and tracks tasks (minimal, honest implementation)."""

from __future__ import annotations

from typing import Any

from app.core.exceptions import JARError
from app.core.identifiers import utc_now
from app.runtime.models import Task, TaskStatus

__all__ = ["TaskManager"]


class TaskManager:
    """In-memory task tracker.

    Only trivial, honest bookkeeping is implemented here. Real scheduling and
    orchestration belong to later phases.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}

    def create_task(
        self,
        *,
        input: Any = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Task:
        task = Task(
            input=input,
            session_id=session_id,
            agent_id=agent_id,
            metadata=metadata or {},
        )
        self._tasks[task.task_id] = task
        return task

    def register(self, task: Task) -> Task:
        """Track an already-constructed task."""
        self._tasks[task.task_id] = task
        return task

    def get(self, task_id: str) -> Task:
        task = self._tasks.get(task_id)
        if task is None:
            raise JARError(f"Task not found: {task_id}")
        return task

    def find(self, task_id: str) -> Task | None:
        """Return the task if tracked, else None (non-raising lookup)."""
        return self._tasks.get(task_id)

    def list(self) -> list[Task]:
        return list(self._tasks.values())

    def update_status(self, task_id: str, status: TaskStatus) -> Task:
        task = self.get(task_id)
        task = task.model_copy(update={"status": status, "updated_at": utc_now()})
        self._tasks[task_id] = task
        return task

    def complete(self, task_id: str, result: Any = None) -> Task:
        return self.update_status(task_id, TaskStatus.COMPLETED).model_copy(
            update={"result": result, "updated_at": utc_now()}
        )

    def fail(self, task_id: str, error: str) -> Task:
        return self.update_status(task_id, TaskStatus.FAILED).model_copy(
            update={"error": error, "updated_at": utc_now()}
        )

    def __len__(self) -> int:
        return len(self._tasks)
