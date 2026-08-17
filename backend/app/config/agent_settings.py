"""Agent orchestration configuration (Phase 7).

The agent orchestration layer is local-first and degrades gracefully when
the LLM is unavailable (deterministic routing/builder fallback). Dynamic
agent creation is opt-in and bounded so the system cannot be flooded with
auto-created agents.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["AgentSettings"]


class AgentSettings(BaseSettings):
    """Agent orchestration subsystem configuration."""

    # --- Master switch ---
    enabled: bool = Field(default=True, alias="AGENT_ENABLED")

    # --- Routing ---
    # Use the LLM for semantic task→agent classification. When disabled or
    # when the LLM is unavailable, deterministic capability-based routing is
    # used. The system is never unusable because of LLM unavailability.
    llm_routing: bool = Field(default=True, alias="AGENT_LLM_ROUTING")

    # --- Dynamic agent builder ---
    dynamic_agents_enabled: bool = Field(default=True, alias="AGENT_DYNAMIC_ENABLED")
    # Hard cap on dynamically-created definitions (prevents runaway growth).
    max_dynamic_agents: int = Field(default=50, alias="AGENT_MAX_DYNAMIC")
    # CRITICAL-risk dynamic agents never auto-activate (always rejected).
    # HIGH-risk dynamic agents require manual activation.
    auto_activate_dynamic: bool = Field(default=True, alias="AGENT_AUTO_ACTIVATE_DYNAMIC")

    # --- Evaluation ---
    # Use the LLM as an evaluator/judge. Objective metrics are always computed
    # from evidence; this only controls whether an LLM judgment is added.
    llm_evaluation: bool = Field(default=True, alias="AGENT_LLM_EVALUATION")
    # Minimum evidence samples before a lifecycle retirement recommendation
    # is considered. A single low-usage sample never retires an agent.
    min_samples_for_retire: int = Field(default=5, alias="AGENT_MIN_SAMPLES_RETIRE")
    # Recent-usage window (number of most recent executions) for the
    # lifecycle manager's "recent usage" signal.
    recent_window: int = Field(default=10, alias="AGENT_RECENT_WINDOW")

    # --- Memory integration ---
    # Inject relevant memory context into agent dispatch (Phase 6 memory).
    memory_aware: bool = Field(default=True, alias="AGENT_MEMORY_AWARE")
    # Persist a memory of meaningful agent executions (not every trivial one).
    persist_execution_memory: bool = Field(default=True, alias="AGENT_PERSIST_MEMORY")

    # --- Persistence ---
    pg_persistence: bool = Field(default=True, alias="AGENT_PG_PERSISTENCE")

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="AGENT_", extra="ignore", frozen=True, populate_by_name=True
    )
