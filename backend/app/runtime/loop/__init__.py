"""Universal Loop Engine.

The execution foundation for all JAR-63 workflows::

    DISCOVER → PLAN → EXECUTE → VERIFY → DECIDE → ITERATE

The loop engine is infrastructure, not an intelligent agent. Stage
implementations provide behavior; the controller orchestrates stages,
maintains state, evaluates stop conditions, publishes events, and enforces
policy. Future agents (LLM-powered or deterministic) plug in through the
stage and verifier interfaces without changing the core engine.
"""

from app.runtime.loop.conditions import (
    CancellationCondition,
    FailureCondition,
    MaxIterationsCondition,
    StopCondition,
    StopDecision,
    SuccessCondition,
    TimeoutCondition,
    default_stop_conditions,
)
from app.runtime.loop.loop_context import LoopContext
from app.runtime.loop.loop_controller import LoopController
from app.runtime.loop.loop_errors import (
    LoopCancelledError,
    LoopError,
    LoopMaxIterationsError,
    LoopPolicyError,
    LoopStageError,
    LoopTimeoutError,
    VerificationError,
)
from app.runtime.loop.loop_policy import LoopPolicy
from app.runtime.loop.loop_result import LoopFinalStatus, LoopResult
from app.runtime.loop.loop_state import (
    ActionType,
    ExecutionResult,
    IterationRecord,
    LoopState,
    LoopStatus,
    NextAction,
    PlanStep,
    StageStatus,
)
from app.runtime.loop.stages import (
    DefaultDiscoverStage,
    DefaultExecuteStage,
    DefaultIterateStage,
    DefaultPlanStage,
    DefaultVerifyStage,
    DiscoveryResult,
    ExecuteStage,
    IterateStage,
    LoopStage,
    PlanResult,
    PlanStage,
    VerifyStage,
)
from app.runtime.loop.verification import (
    CallableVerifier,
    CompositeVerifier,
    ExactMatchVerifier,
    VerificationEvidence,
    VerificationResult,
    VerificationStatus,
    Verifier,
)

__all__ = [
    "ActionType",
    "CallableVerifier",
    "CancellationCondition",
    "CompositeVerifier",
    "DefaultDiscoverStage",
    "DefaultExecuteStage",
    "DefaultIterateStage",
    "DefaultPlanStage",
    "DefaultVerifyStage",
    "DiscoveryResult",
    "DiscoverStage",
    "ExactMatchVerifier",
    "ExecuteStage",
    "ExecutionResult",
    "FailureCondition",
    "IterateStage",
    "IterationRecord",
    "LoopCancelledError",
    "LoopContext",
    "LoopController",
    "LoopError",
    "LoopFinalStatus",
    "LoopMaxIterationsError",
    "LoopPolicy",
    "LoopPolicyError",
    "LoopResult",
    "LoopStage",
    "LoopStageError",
    "LoopState",
    "LoopStatus",
    "LoopTimeoutError",
    "MaxIterationsCondition",
    "NextAction",
    "PlanResult",
    "PlanStage",
    "PlanStep",
    "StageStatus",
    "StopCondition",
    "StopDecision",
    "SuccessCondition",
    "TimeoutCondition",
    "VerificationError",
    "VerificationEvidence",
    "VerificationResult",
    "VerificationStatus",
    "Verifier",
    "VerifyStage",
    "default_stop_conditions",
]
