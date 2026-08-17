"""Agent catalog: the persistent agent-definition registry (Phase 7).

The :class:`AgentDefinitionRegistry` (catalog) is the central lookup for
*persistent agent definitions*. It sits alongside the existing
:class:`~app.agents.registry.AgentRegistry` (which holds *runtime instances*):
definitions persist and describe agents; runtime instances are constructed
from definitions by the dispatcher and may be ephemeral.

The catalog is the single place the router/dispatcher query for definitions.
Other modules must not instantiate agent definitions ad hoc — they go through
the catalog so lifecycle, versioning, and usage tracking stay consistent.

Persistence is delegated to an :class:`AgentDefinitionStore`; an optional
in-memory overlay (:class:`InMemoryAgentDefinitionStore`) is provided for
tests and demos that want to avoid the database.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.agents.definitions import (
    AgentDefinition,
    AgentLifecycleState,
    AgentRecommendation,
    AgentUsageMetadata,
    AgentVersion,
)
from app.agents.interface import AgentCapability
from app.core.exceptions import (
    AgentDefinitionAlreadyExistsError,
    AgentDefinitionNotFoundError,
    AgentDefinitionValidationError,
)
from app.core.identifiers import utc_now
from app.core.logging import get_logger
from app.events import Event, EventBus, EventType

if TYPE_CHECKING:
    from app.agents.store import AgentDefinitionStore

logger = get_logger("agents.catalog")

__all__ = ["AgentDefinitionRegistry", "InMemoryAgentDefinitionStore"]


class InMemoryAgentDefinitionStore:
    """In-memory definition store for tests/demos (no DB required).

    Implements the same methods as :class:`AgentDefinitionStore`.
    """

    def __init__(self) -> None:
        self._defs: dict[str, AgentDefinition] = {}

    async def upsert(self, definition: AgentDefinition) -> AgentDefinition:
        self._defs[definition.agent_id] = definition
        return definition

    async def get(self, agent_id: str) -> AgentDefinition | None:
        return self._defs.get(agent_id)

    async def list(
        self,
        *,
        lifecycle: AgentLifecycleState | None = None,
        limit: int = 200,
    ) -> list[AgentDefinition]:
        items = list(self._defs.values())
        if lifecycle is not None:
            items = [d for d in items if d.lifecycle is lifecycle]
        items = sorted(items, key=lambda d: d.created_at, reverse=True)
        return items[:limit]

    async def list_active(self, limit: int = 200) -> list[AgentDefinition]:
        return await self.list(lifecycle=AgentLifecycleState.ACTIVE, limit=limit)

    async def set_lifecycle(
        self, agent_id: str, lifecycle: AgentLifecycleState
    ) -> AgentDefinition | None:
        d = self._defs.get(agent_id)
        if d is None:
            return None
        updated = d.model_copy(update={"lifecycle": lifecycle, "updated_at": utc_now()})
        self._defs[agent_id] = updated
        return updated

    async def update_usage(
        self, agent_id: str, usage: AgentUsageMetadata
    ) -> AgentDefinition | None:
        d = self._defs.get(agent_id)
        if d is None:
            return None
        updated = d.model_copy(update={"usage": usage, "updated_at": utc_now()})
        self._defs[agent_id] = updated
        return updated

    async def update(self, agent_id: str, **fields: object) -> AgentDefinition | None:
        d = self._defs.get(agent_id)
        if d is None:
            return None
        allowed = {
            "name",
            "description",
            "purpose",
            "instructions",
            "constraints",
            "verification_strategy",
            "required_tools",
            "preferred_tools",
            "task_types",
            "model_requirements",
            "privacy",
            "risk",
            "version",
            "auto_activate",
            "metadata",
        }
        clean = {k: v for k, v in fields.items() if k in allowed}
        if not clean:
            return d
        clean["updated_at"] = utc_now()
        updated = d.model_copy(update=clean)
        self._defs[agent_id] = updated
        return updated

    async def delete(self, agent_id: str) -> bool:
        return self._defs.pop(agent_id, None) is not None

    async def count(self) -> int:
        return len(self._defs)


class AgentDefinitionRegistry:
    """Central registry of persistent agent definitions.

    Wraps a store and publishes lifecycle events on transitions. Keeps the
    router/dispatcher from reaching into arbitrary modules for definitions.
    """

    def __init__(
        self,
        store: AgentDefinitionStore | InMemoryAgentDefinitionStore,
        *,
        event_bus: EventBus | None = None,
    ) -> None:
        self._store = store
        self._event_bus = event_bus

    @property
    def store(self) -> AgentDefinitionStore | InMemoryAgentDefinitionStore:
        return self._store

    async def register(self, definition: AgentDefinition) -> AgentDefinition:
        """Register a new definition. Rejects duplicate ids."""
        existing = await self._store.get(definition.agent_id)
        if existing is not None:
            raise AgentDefinitionAlreadyExistsError(definition.agent_id)
        _validate_definition(definition)
        await self._store.upsert(definition)
        await self._publish(EventType.AGENT_CREATED, definition)
        logger.bind(event="agent.definition.registered", agent_id=definition.agent_id).info(
            "Registered agent definition '{}'", definition.name
        )
        return definition

    async def upsert(self, definition: AgentDefinition) -> AgentDefinition:
        """Insert or replace a definition (used by dynamic builder/versioning)."""
        _validate_definition(definition)
        existing = await self._store.get(definition.agent_id)
        await self._store.upsert(definition)
        await self._publish(
            EventType.AGENT_UPDATED if existing is not None else EventType.AGENT_CREATED,
            definition,
        )
        return definition

    async def get(self, agent_id: str) -> AgentDefinition:
        d = await self._store.get(agent_id)
        if d is None:
            raise AgentDefinitionNotFoundError(agent_id)
        return d

    async def find(self, agent_id: str) -> AgentDefinition | None:
        return await self._store.get(agent_id)

    async def list(
        self,
        *,
        lifecycle: AgentLifecycleState | None = None,
        limit: int = 200,
    ) -> list[AgentDefinition]:
        return await self._store.list(lifecycle=lifecycle, limit=limit)

    async def list_active(self, limit: int = 200) -> list[AgentDefinition]:
        return await self._store.list_active(limit=limit)

    async def find_by_capability(
        self, capability: AgentCapability, *, active_only: bool = True
    ) -> list[AgentDefinition]:
        items = await self._store.list_active() if active_only else await self._store.list()
        return [d for d in items if capability in d.capabilities]

    async def find_by_task_type(
        self, task_type: str, *, active_only: bool = True
    ) -> list[AgentDefinition]:
        items = await self._store.list_active() if active_only else await self._store.list()
        return [d for d in items if task_type in d.task_types]

    async def update_metadata(self, agent_id: str, **fields: object) -> AgentDefinition:
        d = await self._store.update(agent_id, **fields)
        if d is None:
            raise AgentDefinitionNotFoundError(agent_id)
        await self._publish(EventType.AGENT_UPDATED, d)
        return d

    async def update_usage(
        self, agent_id: str, usage: AgentUsageMetadata
    ) -> AgentDefinition | None:
        return await self._store.update_usage(agent_id, usage)

    async def set_lifecycle(
        self,
        agent_id: str,
        lifecycle: AgentLifecycleState,
        *,
        reason: str = "",
    ) -> AgentDefinition:
        d = await self._store.set_lifecycle(agent_id, lifecycle)
        if d is None:
            raise AgentDefinitionNotFoundError(agent_id)
        event_type = {
            AgentLifecycleState.ACTIVE: EventType.AGENT_UPDATED,
            AgentLifecycleState.INACTIVE: EventType.AGENT_UPDATED,
            AgentLifecycleState.DEPRECATED: EventType.AGENT_DEPRECATED,
            AgentLifecycleState.RETIRED: EventType.AGENT_RETIRED,
        }[lifecycle]
        await self._publish(event_type, d, extra={"reason": reason})
        logger.bind(event="agent.lifecycle", agent_id=agent_id, lifecycle=lifecycle.value).info(
            "Agent '{}' lifecycle -> {}", agent_id, lifecycle.value
        )
        return d

    async def deactivate(self, agent_id: str, *, reason: str = "") -> AgentDefinition:
        return await self.set_lifecycle(agent_id, AgentLifecycleState.INACTIVE, reason=reason)

    async def deprecate(self, agent_id: str, *, reason: str = "") -> AgentDefinition:
        return await self.set_lifecycle(agent_id, AgentLifecycleState.DEPRECATED, reason=reason)

    async def retire(self, agent_id: str, *, reason: str = "") -> AgentDefinition:
        return await self.set_lifecycle(agent_id, AgentLifecycleState.RETIRED, reason=reason)

    async def activate(self, agent_id: str, *, reason: str = "") -> AgentDefinition:
        return await self.set_lifecycle(agent_id, AgentLifecycleState.ACTIVE, reason=reason)

    async def create_version(
        self,
        agent_id: str,
        *,
        version: str,
        reason: str = "",
        updates: dict | None = None,
    ) -> AgentDefinition:
        """Create a new version of a definition, preserving the prior version.

        The new version replaces the live definition (same agent_id, new
        ``version``) so the router picks it up. Execution records keep
        referencing the version they ran under, so history stays interpretable.
        """
        d = await self.get(agent_id)
        stamp = AgentVersion(version=version, reason=reason)
        _validate_version_progression(d.version, stamp.version)
        change = {"version": stamp.version, "updated_at": utc_now()}
        if updates:
            change.update(updates)
        updated = d.model_copy(update=change)
        await self._store.upsert(updated)
        await self._publish(
            EventType.AGENT_VERSION_CREATED,
            updated,
            extra={"from_version": d.version, "to_version": stamp.version, "reason": reason},
        )
        logger.bind(event="agent.version", agent_id=agent_id, version=stamp.version).info(
            "Agent '{}' version {} -> {}", agent_id, d.version, stamp.version
        )
        return updated

    async def record_recommendation(
        self, agent_id: str, recommendation: AgentRecommendation, *, reason: str = ""
    ) -> AgentDefinition | None:
        """Apply a lifecycle recommendation to the definition state.

        Returns the updated definition for DEACTIVATE/RETIRE, or ``None`` for
        KEEP/IMPROVE (informational, no state change).
        """
        if recommendation is AgentRecommendation.KEEP:
            return None
        if recommendation is AgentRecommendation.IMPROVE:
            # Keep active; the recommendation is informational.
            return None
        if recommendation is AgentRecommendation.DEACTIVATE:
            return await self.deactivate(agent_id, reason=reason)
        if recommendation is AgentRecommendation.RETIRE:
            return await self.retire(agent_id, reason=reason)
        return None

    async def count(self) -> int:
        return await self._store.count()

    async def _publish(
        self,
        event_type: EventType,
        definition: AgentDefinition,
        *,
        extra: dict | None = None,
    ) -> None:
        if self._event_bus is None:
            return
        payload = {
            "agent_id": definition.agent_id,
            "name": definition.name,
            "version": definition.version,
            "lifecycle": definition.lifecycle.value,
            "kind": definition.kind.value,
            "dynamic": definition.dynamic,
        }
        if extra:
            payload.update(extra)
        await self._event_bus.publish(
            Event.create(
                event_type,
                agent_id=definition.agent_id,
                payload=payload,
                metadata={"agent_id": definition.agent_id, "version": definition.version},
            )
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_FORBIDDEN_TOOL_NAMES = {"shell", "exec", "eval", "subprocess", "python_exec"}


def _validate_definition(definition: AgentDefinition) -> None:
    """Reject malformed/unsafe definitions before they enter the catalog."""
    if not definition.agent_id.strip():
        raise AgentDefinitionValidationError("agent_id must not be empty")
    if not definition.name.strip():
        raise AgentDefinitionValidationError("name must not be empty")
    # Dynamic agents never auto-elevate to CRITICAL.
    if definition.dynamic and definition.risk.value == "critical":
        raise AgentDefinitionValidationError("dynamic agents may not declare CRITICAL risk")
    # Required tools must not reference unrestricted execution surfaces.
    bad = _FORBIDDEN_TOOL_NAMES.intersection(
        set(definition.required_tools) | set(definition.preferred_tools)
    )
    if bad:
        raise AgentDefinitionValidationError(
            f"definition references unrestricted tool(s): {sorted(bad)}"
        )
    # Instructions must not request raw code execution of the agent itself.
    lowered = definition.instructions.lower()
    if "import os" in lowered and "subprocess" in lowered:
        raise AgentDefinitionValidationError(
            "instructions must not request os+subprocess execution"
        )


def _validate_version_progression(old: str, new: str) -> None:
    """Reject non-monotonic version changes (best-effort semantic check)."""
    if old == new:
        raise AgentDefinitionValidationError(
            f"new version '{new}' must differ from current '{old}'"
        )
