"""Loop context: the dependency-injected environment for a loop run.

``LoopContext`` provides controlled, read-mostly access to everything a stage
needs: the loop state, the task, the registries, the event bus, and runtime
services. It carries no global mutable state — every collaborator is injected
at construction.

Stages receive the context and return typed results; they never reach outside
it for dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.agents.registry import AgentRegistry
from app.config import Settings
from app.events import EventBus
from app.runtime.loop.loop_policy import LoopPolicy
from app.runtime.loop.loop_state import LoopState
from app.runtime.models import Task
from app.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from app.runtime.session_manager import SessionManager

__all__ = ["LoopContext"]


@dataclass
class LoopContext:
    """Injected dependencies + the live loop state for one loop run."""

    state: LoopState
    task: Task
    agent_registry: AgentRegistry
    tool_registry: ToolRegistry
    event_bus: EventBus
    policy: LoopPolicy = field(default_factory=LoopPolicy)
    settings: Settings | None = None
    session_manager: SessionManager | None = None
    # Optional configuration bag for stage-specific deterministic behavior.
    # Kept as a plain dict because it is opaque stage configuration, not
    # primary loop state.
    stage_config: dict[str, object] = field(default_factory=dict)

    @property
    def loop_id(self) -> str:
        return self.state.loop_id

    @property
    def task_id(self) -> str:
        return self.state.task_id

    @property
    def session_id(self) -> str | None:
        return self.state.session_id

    def update_state(self, **updates: object) -> LoopState:
        """Evolve the live state and return the new snapshot."""
        self.state = self.state.evolve(**updates)
        return self.state
