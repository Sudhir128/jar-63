"""Agent lifecycle manager (Phase 7).

Recommends lifecycle transitions from *multiple evidence dimensions* and
applies them via the catalog. This is deliberately NOT a predictive ML model
— it is a transparent evidence + evaluator architecture. An agent is never
retired on a single low-usage sample; the ``min_samples_for_retire`` setting
enforces a minimum evidence base before retirement is considered.

Evidence dimensions considered:
* usage frequency (dispatch count, recent usage)
* success rate
* verification quality (success implies verified evidence)
* execution latency
* failure rate
* capability overlap (is another active agent superseding this one?)
* recent usage
* evaluator assessment

Recommendations:
    ACTIVE → IMPROVE → ACTIVE
    ACTIVE → DEPRECATE → RETIRE

Retirement preserves historical records (the catalog never hard-deletes; it
transitions the lifecycle state).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.agents.definitions import AgentDefinition, AgentRecommendation
from app.agents.evaluation import compute_objective_metrics
from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.agents.catalog import AgentDefinitionRegistry
    from app.agents.evaluation import AgentEvaluator
    from app.agents.store import AgentExecutionStore

logger = get_logger("agents.lifecycle")

__all__ = ["LifecycleManager", "LifecycleAdvice"]


@dataclass(frozen=True)
class LifecycleAdvice:
    """A lifecycle recommendation for a definition."""

    agent_id: str
    recommendation: AgentRecommendation
    reason: str
    evidence: dict[str, Any]


class LifecycleManager:
    """Recommends and applies lifecycle transitions from evidence."""

    def __init__(
        self,
        catalog: AgentDefinitionRegistry,
        execution_store: AgentExecutionStore,
        *,
        evaluator: AgentEvaluator | None = None,
        min_samples_for_retire: int = 5,
        recent_window: int = 10,
    ) -> None:
        self._catalog = catalog
        self._exec_store = execution_store
        self._evaluator = evaluator
        self._min_samples_for_retire = min_samples_for_retire
        self._recent_window = recent_window

    async def advise(self, definition: AgentDefinition) -> LifecycleAdvice:
        """Compute a lifecycle recommendation from evidence dimensions."""
        records = await self._exec_store.list_for_agent(
            definition.agent_id, limit=max(self._recent_window, self._min_samples_for_retire)
        )
        metrics = compute_objective_metrics(records)
        recent = records[: self._recent_window]
        recent_metrics = compute_objective_metrics(recent)
        overlap = await self._capability_overlap(definition)

        usage = definition.usage
        success_rate = metrics.get("success_rate", 0.0)
        sample_count = metrics.get("sample_count", 0)
        recent_count = recent_metrics.get("sample_count", 0)
        policy_violations = metrics.get("policy_violation_count", 0)

        # Start from the evaluator's recommendation (if available).
        base_rec = AgentRecommendation.KEEP
        if self._evaluator is not None:
            evaluation = await self._evaluator.evaluate(definition, records=records)
            base_rec = evaluation.recommendation

        reason_parts: list[str] = []
        rec = base_rec

        # Evidence rules (transparent, not an ML model):
        if policy_violations > 0:
            reason_parts.append(f"{policy_violations} policy violation(s)")
            rec = AgentRecommendation.DEACTIVATE
        if sample_count == 0 and usage.dispatch_count == 0:
            reason_parts.append("no usage evidence; keep active")
            rec = AgentRecommendation.KEEP
        elif success_rate < 0.3 and sample_count >= 3:
            reason_parts.append(f"low success rate ({success_rate:.0%})")
            if rec is AgentRecommendation.KEEP:
                rec = AgentRecommendation.IMPROVE

        # Retirement: only with a sufficient evidence base, low recent usage,
        # and a superseding active agent (capability overlap).
        if sample_count >= self._min_samples_for_retire and recent_count == 0 and overlap:
            reason_parts.append(
                f"no recent usage in last {self._recent_window} and superseded by {overlap}"
            )
            rec = AgentRecommendation.RETIRE
        # Never retire below the minimum sample floor.
        if rec is AgentRecommendation.RETIRE and sample_count < self._min_samples_for_retire:
            rec = AgentRecommendation.DEPRECATE
            reason_parts.append(
                f"insufficient samples for retire ({sample_count}<{self._min_samples_for_retire}); deprecate"
            )

        if not reason_parts:
            reason_parts.append("evidence supports keeping active")
        return LifecycleAdvice(
            agent_id=definition.agent_id,
            recommendation=rec,
            reason="; ".join(reason_parts),
            evidence={
                "sample_count": sample_count,
                "success_rate": success_rate,
                "recent_usage": recent_count,
                "policy_violations": policy_violations,
                "capability_overlap": overlap,
                "dispatch_count": usage.dispatch_count,
                "base_recommendation": base_rec.value,
            },
        )

    async def apply(self, advice: LifecycleAdvice) -> AgentDefinition | None:
        """Apply a lifecycle recommendation (KEEP/IMPROVE are no-ops)."""
        if advice.recommendation in (AgentRecommendation.KEEP, AgentRecommendation.IMPROVE):
            return await self._catalog.find(advice.agent_id)
        return await self._catalog.record_recommendation(
            advice.agent_id, advice.recommendation, reason=advice.reason
        )

    async def review_active(self) -> list[LifecycleAdvice]:
        """Review all active definitions and return advice for each."""
        active = await self._catalog.list_active()
        return [await self.advise(d) for d in active]

    async def _capability_overlap(self, definition: AgentDefinition) -> str | None:
        """Return the agent_id of an active agent superseding this one, or None."""
        if not definition.capabilities:
            return None
        others = await self._catalog.list_active()
        for other in others:
            if other.agent_id == definition.agent_id:
                continue
            # Only count as superseding if this agent's caps are a subset of
            # another's, and the other is strictly a superset or equal but
            # registered earlier (smaller id wins as a deterministic tie-break).
            if (
                definition.capabilities.issubset(other.capabilities)
                and other.created_at <= definition.created_at
                and other.agent_id < definition.agent_id
            ):
                return other.agent_id
        return None
