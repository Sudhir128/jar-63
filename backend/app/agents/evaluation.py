"""Agent evaluation: evidence-based, not a fake ML formula (Phase 7).

Evaluation collects **objective execution evidence** (success, verification,
iterations, tool failures, latency, retries, cancellations, policy
violations, memory relevance, model failures) and feeds it through an
**evaluator abstraction**. The LLM evaluator is a *judge*, never the sole
source of truth — objective metrics are computed independently and kept
separate from the judgment.

Architecture::

    Execution Evidence (AgentExecutionRecord)
            ↓
    Objective Metrics (computed facts)
            ↓
    LLM Evaluator / Evaluation Strategy (judge)
            ↓
    Structured AgentEvaluation
            ↓
    Performance History

There is deliberately NO hard-coded weighted formula presented as a "model".
Objective metrics are descriptive statistics; the recommendation is a judgment
by the evaluator grounded in those metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.agents.definitions import (
    AgentDefinition,
    AgentEvaluation,
    AgentPerformanceSummary,
    AgentRecommendation,
)
from app.core.logging import get_logger
from app.events import Event, EventBus, EventType

if TYPE_CHECKING:
    from app.agents.store import AgentEvaluationStore, AgentExecutionStore
    from app.llm.router import ModelRouter

logger = get_logger("agents.evaluation")

__all__ = [
    "EvaluationStrategy",
    "DeterministicEvaluator",
    "LLMEvaluator",
    "AgentEvaluator",
    "compute_objective_metrics",
    "compute_performance_summary",
]


# ---------------------------------------------------------------------------
# Objective metrics (facts, not judgments)
# ---------------------------------------------------------------------------


def compute_objective_metrics(
    records: list[Any],
) -> dict[str, Any]:
    """Compute objective metrics from execution records (measured facts).

    No recommendation is produced here — only descriptive statistics. These
    are facts that the evaluator judges.
    """
    if not records:
        return {
            "sample_count": 0,
            "success_count": 0,
            "success_rate": 0.0,
            "avg_iterations": 0.0,
            "avg_latency_ms": 0.0,
            "tool_failure_count": 0,
            "tool_failure_rate": 0.0,
            "policy_violation_count": 0,
            "policy_violation_rate": 0.0,
            "cancellation_count": 0,
            "cancellation_rate": 0.0,
            "total_tool_calls": 0,
            "total_retries": 0,
            "model_failure_count": 0,
        }
    n = len(records)
    success = sum(1 for r in records if r.success)
    iterations = sum(r.iterations_used for r in records) / n
    latencies = [r.latency_ms for r in records if r.latency_ms is not None]
    avg_latency = (sum(latencies) / len(latencies)) if latencies else 0.0
    tool_failures = sum(r.tool_failures for r in records)
    tool_calls = sum(r.tool_calls for r in records)
    policy_violations = sum(r.policy_violations for r in records)
    cancellations = sum(1 for r in records if r.cancelled)
    retries = sum(r.retries for r in records)
    model_failures = sum(r.model_failures for r in records)
    return {
        "sample_count": n,
        "success_count": success,
        "success_rate": success / n,
        "avg_iterations": iterations,
        "avg_latency_ms": avg_latency,
        "tool_failure_count": tool_failures,
        "tool_failure_rate": (tool_failures / tool_calls) if tool_calls else 0.0,
        "policy_violation_count": policy_violations,
        "policy_violation_rate": (policy_violations / n),
        "cancellation_count": cancellations,
        "cancellation_rate": cancellations / n,
        "total_tool_calls": tool_calls,
        "total_retries": retries,
        "model_failure_count": model_failures,
    }


def compute_performance_summary(
    agent_id: str, version: str, records: list[Any]
) -> AgentPerformanceSummary:
    """Aggregate records into a typed performance summary."""
    m = compute_objective_metrics(records)
    return AgentPerformanceSummary(
        agent_id=agent_id,
        agent_version=version,
        sample_count=m["sample_count"],
        success_rate=m["success_rate"],
        avg_iterations=m["avg_iterations"],
        avg_latency_ms=m["avg_latency_ms"],
        tool_failure_rate=m["tool_failure_rate"],
        policy_violation_rate=m["policy_violation_rate"],
        cancellation_rate=m["cancellation_rate"],
        recent_dispatch_count=m["sample_count"],
    )


# ---------------------------------------------------------------------------
# Evaluation strategies
# ---------------------------------------------------------------------------


class EvaluationStrategy:
    """Abstract evaluator: produces a judgment from objective metrics."""

    def evaluate(
        self,
        definition: AgentDefinition,
        metrics: dict[str, Any],
        records: list[Any],
    ) -> AgentEvaluation:
        raise NotImplementedError


class DeterministicEvaluator(EvaluationStrategy):
    """Rule-grounded evaluator with no LLM.

    Produces a transparent judgment from objective metrics. The rules are
    intentionally interpretable (not a hidden weighted "ML" formula): they
    enumerate the evidence dimensions and a clear recommendation. A single
    low-usage sample never produces a RETIRE recommendation — that is the
    lifecycle manager's policy, enforced via ``min_samples_for_retire``.
    """

    def evaluate(
        self,
        definition: AgentDefinition,
        metrics: dict[str, Any],
        records: list[Any],
    ) -> AgentEvaluation:
        strengths: list[str] = []
        weaknesses: list[str] = []
        failure_reasons: list[str] = []
        rec = AgentRecommendation.KEEP

        success_rate = metrics.get("success_rate", 0.0)
        sample_count = metrics.get("sample_count", 0)
        tool_failure_rate = metrics.get("tool_failure_rate", 0.0)
        cancellation_rate = metrics.get("cancellation_rate", 0.0)

        if success_rate >= 0.8:
            strengths.append(f"High success rate ({success_rate:.0%}).")
        elif success_rate >= 0.5:
            weaknesses.append(f"Moderate success rate ({success_rate:.0%}).")
        else:
            weaknesses.append(f"Low success rate ({success_rate:.0%}).")
            failure_reasons.append("frequent verification/execution failures")

        if tool_failure_rate > 0.3:
            weaknesses.append("High tool failure rate.")
            failure_reasons.append("tool execution failures")
        if policy_violations := metrics.get("policy_violation_count", 0):
            weaknesses.append(f"{policy_violations} policy violation(s).")
            failure_reasons.append("policy violations")
        if cancellation_rate > 0.3:
            weaknesses.append("High cancellation rate.")

        # Recommendation judgment grounded in evidence (NOT a fake prediction).
        if sample_count == 0:
            assessment = "No execution evidence yet; keeping active."
            rec = AgentRecommendation.KEEP
        elif policy_violations > 0 or success_rate < 0.3:
            assessment = "Agent is underperforming or violating policy; deactivate."
            rec = AgentRecommendation.DEACTIVATE
        elif success_rate < 0.6 or cancellation_rate > 0.3:
            assessment = "Agent needs improvement; keep active but flag."
            rec = AgentRecommendation.IMPROVE
        else:
            assessment = "Agent is performing reliably."
            rec = AgentRecommendation.KEEP

        return AgentEvaluation(
            agent_id=definition.agent_id,
            agent_version=definition.version,
            objective_metrics=metrics,
            overall_assessment=assessment,
            strengths=strengths,
            weaknesses=weaknesses,
            failure_reasons=failure_reasons,
            capability_assessment={
                cap.value: {"success_rate": success_rate} for cap in definition.capabilities
            },
            recommended_changes=weaknesses,
            confidence=min(1.0, 0.5 + 0.1 * min(sample_count, 5)),
            recommendation=rec,
            evaluator="deterministic",
        )


class LLMEvaluator(EvaluationStrategy):
    """LLM-as-judge evaluator.

    Uses the LLM to produce a structured assessment grounded in the objective
    metrics (passed to the model). Falls back to :class:`DeterministicEvaluator`
    whenever the LLM is unavailable or returns invalid output — the system is
    never unusable because of LLM unavailability. The LLM never sees secrets;
    it sees only objective metrics (counts/rates), never prompt/memory content.
    """

    def __init__(
        self,
        *,
        model_router: ModelRouter | None = None,
        fallback: DeterministicEvaluator | None = None,
    ) -> None:
        self._router = model_router
        self._fallback = fallback or DeterministicEvaluator()

    async def evaluate_async(
        self,
        definition: AgentDefinition,
        metrics: dict[str, Any],
        records: list[Any],
    ) -> AgentEvaluation:
        if self._router is None:
            return self._fallback.evaluate(definition, metrics, records)
        client = await self._select_client()
        if client is None:
            return self._fallback.evaluate(definition, metrics, records)
        try:
            from app.llm.models import LLMMessage, LLMRequest, MessageRole

            messages = [
                LLMMessage(
                    role=MessageRole.SYSTEM,
                    content=(
                        "You are an agent evaluator/judge. You receive OBJECTIVE "
                        "execution metrics (counts and rates only — no secrets, no "
                        "prompt/memory content). Produce a structured assessment and "
                        "a recommendation (keep|improve|deactivate|retire). Respond "
                        'ONLY with JSON: {"overall_assessment": str, "strengths": '
                        '[str], "weaknesses": [str], "recommended_changes": [str], '
                        '"confidence": float, "recommendation": str}.'
                    ),
                ),
                LLMMessage(
                    role=MessageRole.USER,
                    content=(
                        f"Agent: {definition.name} ({definition.agent_id} v{definition.version})\n"
                        f"Purpose: {definition.purpose}\n"
                        f"Objective metrics: {metrics}\n"
                        f"Evaluate this agent."
                    ),
                ),
            ]
            request = LLMRequest(
                model=self._select_model_id(),
                messages=messages,
                temperature=0.1,
                metadata={"stage": "agent_evaluation"},
            )
            response = await client.generate(request)
        except Exception as exc:  # noqa: BLE001 - LLM unavailable → fallback
            logger.bind(event="agent.eval.llm.error", error=type(exc).__name__).debug(
                "LLM evaluation failed (falling back): {}", str(exc)
            )
            return self._fallback.evaluate(definition, metrics, records)
        return self._parse(definition, metrics, response, records)

    async def _select_client(self):
        if self._router is None:
            return None
        try:
            from app.llm.models import ModelCapability, PrivacyLevel
            from app.llm.router import RoutingRequest
        except ImportError:  # pragma: no cover
            return None
        request = RoutingRequest(
            capabilities={ModelCapability.CHAT},
            privacy=PrivacyLevel.INTERNAL,
            prefer_local=True,
            metadata={"stage": "agent_evaluation"},
        )
        try:
            selection = await self._router.select(request)
        except Exception:  # noqa: BLE001
            return None
        providers = self._router._providers  # noqa: SLF001
        if not providers.exists(selection.provider):
            return None
        return providers.get(selection.provider)

    def _select_model_id(self) -> str:
        return ""

    def _parse(
        self,
        definition: AgentDefinition,
        metrics: dict[str, Any],
        response: Any,
        records: list[Any],
    ) -> AgentEvaluation:
        import json

        raw = response.content or ""
        try:
            text = raw.strip()
            if text.startswith("```"):
                text = text.strip("`")
                if text.lower().startswith("json"):
                    text = text[4:]
                text = text.strip("`").strip()
            parsed = json.loads(text)
        except Exception:  # noqa: BLE001
            return self._fallback.evaluate(definition, metrics, records)
        valid_recs = {r.value for r in AgentRecommendation}
        rec_value = parsed.get("recommendation", "keep")
        if rec_value not in valid_recs:
            return self._fallback.evaluate(definition, metrics, records)
        try:
            confidence = float(parsed.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.5
        return AgentEvaluation(
            agent_id=definition.agent_id,
            agent_version=definition.version,
            objective_metrics=metrics,
            overall_assessment=str(parsed.get("overall_assessment", "")),
            strengths=list(parsed.get("strengths", [])),
            weaknesses=list(parsed.get("weaknesses", [])),
            failure_reasons=[
                r.failure_reason for r in records if not r.success and r.failure_reason
            ],
            capability_assessment={
                cap.value: {"success_rate": metrics.get("success_rate", 0.0)}
                for cap in definition.capabilities
            },
            recommended_changes=list(parsed.get("recommended_changes", [])),
            confidence=confidence,
            recommendation=AgentRecommendation(rec_value),
            evaluator="llm",
        )

    def evaluate(
        self,
        definition: AgentDefinition,
        metrics: dict[str, Any],
        records: list[Any],
    ) -> AgentEvaluation:
        # Sync entrypoint: deterministic fallback (LLM path is async).
        return self._fallback.evaluate(definition, metrics, records)


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


@dataclass
class AgentEvaluator:
    """Collect evidence → compute metrics → judge → persist evaluation.

    The evaluator is a judge; objective facts remain separate from the
    judgment (``objective_metrics`` vs ``overall_assessment``).
    """

    execution_store: AgentExecutionStore
    evaluation_store: AgentEvaluationStore
    strategy: EvaluationStrategy
    event_bus: EventBus | None = None
    # When True and the strategy is an LLMEvaluator, use the async LLM path.
    use_llm: bool = False

    async def evaluate(
        self,
        definition: AgentDefinition,
        *,
        records: list[Any] | None = None,
        limit: int = 100,
    ) -> AgentEvaluation:
        if self.event_bus is not None:
            await self.event_bus.publish(
                Event.create(
                    EventType.AGENT_EVALUATION_STARTED,
                    agent_id=definition.agent_id,
                    payload={"version": definition.version},
                    metadata={"agent_id": definition.agent_id},
                )
            )
        if records is None:
            records = await self.execution_store.list_for_agent(definition.agent_id, limit=limit)
        metrics = compute_objective_metrics(records)
        if self.use_llm and isinstance(self.strategy, LLMEvaluator):
            evaluation = await self.strategy.evaluate_async(definition, metrics, records)
        else:
            evaluation = self.strategy.evaluate(definition, metrics, records)
        await self.evaluation_store.add(evaluation)
        if self.event_bus is not None:
            await self.event_bus.publish(
                Event.create(
                    EventType.AGENT_EVALUATION_COMPLETED,
                    agent_id=definition.agent_id,
                    payload={
                        "recommendation": evaluation.recommendation.value,
                        "confidence": evaluation.confidence,
                        "sample_count": metrics.get("sample_count", 0),
                        "evaluator": evaluation.evaluator,
                    },
                    metadata={"agent_id": definition.agent_id, "version": definition.version},
                )
            )
        return evaluation

    async def history(self, agent_id: str, *, limit: int = 50) -> list[AgentEvaluation]:
        return await self.evaluation_store.list_for_agent(agent_id, limit=limit)
