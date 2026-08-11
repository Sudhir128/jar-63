"""Startup bootstrap: registers deterministic demo agents into the runtime.

This wires the demo agents defined in :mod:`app.runtime.loop.demos` into the
running application's :class:`AgentRegistry` so the Universal Loop Engine can
be exercised end-to-end through the REST API without an LLM.

This is the supported startup-time registration mechanism (it uses the
existing ``AgentRegistry.register`` API during application lifespan startup —
it does NOT add a new REST endpoint). Registration is skipped in production
and is collision-safe so it never conflicts with agents registered elsewhere
(e.g. in tests).
"""

from __future__ import annotations

from app.agents.registry import AgentRegistry
from app.config import Settings
from app.core.exceptions import AgentAlreadyRegisteredError
from app.core.logging import get_logger
from app.runtime.loop.demos import build_demo_agents

logger = get_logger("runtime.bootstrap")

__all__ = ["register_demo_agents"]


async def register_demo_agents(registry: AgentRegistry, settings: Settings) -> int:
    """Register demo agents into ``registry`` unless running in production.

    Returns the number of agents registered. Agents whose id is already
    registered are skipped (no error), so this is safe to call even when other
    agents have been registered first.
    """
    if settings.app.is_prod:
        logger.bind(event="bootstrap.skip").info("Skipping demo agent registration in production.")
        return 0

    registered = 0
    for agent in build_demo_agents():
        try:
            await registry.register(agent)
            registered += 1
        except AgentAlreadyRegisteredError:
            logger.bind(agent_id=agent.agent_id, event="bootstrap.skip").debug(
                "Agent '{}' already registered; skipping.", agent.name
            )
    logger.bind(event="bootstrap.done", count=registered).info(
        "Registered {} demo agent(s).", registered
    )
    return registered
