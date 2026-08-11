"""Deterministic LLM routing and planning demonstrations.

These demos exercise the Phase 2 routing/planning logic with fully
deterministic, offline setups (no real Ollama or cloud required). Each demo
builds a small registry/router configuration and verifies the expected
selection or behavior.

Demos:
1. Local model selection (coding task, local model available → Ollama).
2. Cloud fallback (no local coding model, cloud available → cloud).
3. Offline (network down, local model available → local).
4. Offline + no local model → structured ModelUnavailable error.
5. Private task (local + cloud available → local).
6. LLM planner produces a validated structured plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.config import LLMSettings
from app.events import EventBus, InMemoryEventBus
from app.llm.client import HealthStatus, LLMClient, ProviderHealth
from app.llm.errors import ModelUnavailableError
from app.llm.models import (
    LLMRequest,
    LLMResponse,
    ModelCapability,
    ModelDefinition,
    PrivacyLevel,
)
from app.llm.registry import ModelRegistry, ProviderRegistry
from app.llm.router import (
    AlwaysOfflineNetworkChecker,
    ModelRouter,
    NetworkChecker,
    RoutingRequest,
)

__all__ = [
    "StubLLMClient",
    "build_local_coding_model",
    "build_cloud_coding_model",
    "build_router",
    "demo_local_selection",
    "demo_cloud_fallback",
    "demo_offline_local",
    "demo_offline_no_model",
    "demo_private_task",
    "run_all_demos",
]


class StubLLMClient(LLMClient):
    """A fully in-memory LLM client for deterministic demos/tests.

    Returns a canned structured response. No network access.
    """

    def __init__(
        self,
        provider_name: str,
        *,
        structured_payload: dict[str, Any] | None = None,
        content: str = "",
        health: ProviderHealth | None = None,
    ) -> None:
        self._provider = provider_name
        self._structured = structured_payload
        self._content = content
        self._health = health or ProviderHealth(
            status=HealthStatus.AVAILABLE, provider=provider_name
        )

    @property
    def provider_name(self) -> str:
        return self._provider

    async def generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(content=self._content, model=request.model, provider=self._provider)

    async def generate_structured(
        self, request: LLMRequest, spec: Any | None = None
    ) -> LLMResponse:
        return LLMResponse(
            content="{}",
            model=request.model,
            provider=self._provider,
            metadata={"parsed": self._structured or {}},
        )

    async def check_health(self, model: str | None = None) -> ProviderHealth:
        return self._health

    async def get_model_info(self, model: str) -> ModelDefinition | None:
        return None

    async def close(self) -> None:
        pass


def build_local_coding_model(model_id: str = "qwen2.5-coder:7b") -> ModelDefinition:
    return ModelDefinition(
        model_id=model_id,
        provider="ollama",
        display_name=model_id,
        capabilities={ModelCapability.CHAT, ModelCapability.CODING, ModelCapability.REASONING},
        context_window=32768,
        supports_structured_output=True,
        coding_score=0.8,
        local=True,
        enabled=True,
    )


def build_cloud_coding_model(model_id: str = "cloud-coder:32b") -> ModelDefinition:
    return ModelDefinition(
        model_id=model_id,
        provider="openai_compatible",
        display_name=model_id,
        capabilities={
            ModelCapability.CHAT,
            ModelCapability.CODING,
            ModelCapability.REASONING,
            ModelCapability.TOOL_CALLING,
        },
        context_window=128000,
        supports_structured_output=True,
        coding_score=0.9,
        local=False,
        enabled=True,
    )


def _base_llm_settings(*, allow_cloud_private: bool = False) -> LLMSettings:
    return LLMSettings(
        _env_file=None,
        enabled=True,
        default_provider="ollama",
        default_model="qwen2.5-coder:7b",
        routing_policy="local_first",
        request_timeout=60,
        verbose_logging=False,
        allow_cloud_for_private=allow_cloud_private,
        ollama_base_url="http://localhost:11434",
        ollama_default_model="qwen2.5-coder:7b",
        openai_compatible_enabled=True,
        openai_compatible_base_url="http://cloud.example.com",
        openai_compatible_model="cloud-coder:32b",
    )


def build_router(
    *,
    models: list[ModelDefinition],
    providers: list[tuple[str, LLMClient]] | None = None,
    network_checker: NetworkChecker | None = None,
    settings: LLMSettings | None = None,
    event_bus: EventBus | None = None,
) -> ModelRouter:
    model_registry = ModelRegistry()
    for m in models:
        model_registry.register(m)
    provider_registry = ProviderRegistry()
    for pid, client in providers or []:
        provider_registry.register(pid, client)
    return ModelRouter(
        model_registry=model_registry,
        provider_registry=provider_registry,
        settings=settings or _base_llm_settings(),
        network_checker=network_checker,
        event_bus=event_bus,
    )


@dataclass
class DemoResult:
    name: str
    passed: bool
    detail: str


def demo_local_selection() -> DemoResult:
    """Demo 1: coding task + local model available → Ollama selected."""
    router = build_router(
        models=[build_local_coding_model()],
        providers=[("ollama", StubLLMClient("ollama"))],
    )
    selection = asyncio_run(
        router.select(
            RoutingRequest(capabilities={ModelCapability.CODING}, privacy=PrivacyLevel.PUBLIC)
        )
    )
    ok = (
        selection.provider == "ollama"
        and selection.local
        and "qwen2.5-coder:7b" in selection.model_id
    )
    return DemoResult(
        "demo_local_selection", ok, f"selected={selection.display} local={selection.local}"
    )


def demo_cloud_fallback() -> DemoResult:
    """Demo 2: no local coding model + cloud available → cloud selected."""

    class Online(NetworkChecker):
        async def is_available(self, provider_id: str) -> bool:
            return True

    router = build_router(
        models=[build_cloud_coding_model()],
        providers=[("openai_compatible", StubLLMClient("openai_compatible"))],
        network_checker=Online(),
    )
    selection = asyncio_run(
        router.select(
            RoutingRequest(capabilities={ModelCapability.CODING}, privacy=PrivacyLevel.PUBLIC)
        )
    )
    ok = selection.provider == "openai_compatible" and not selection.local
    return DemoResult(
        "demo_cloud_fallback", ok, f"selected={selection.display} local={selection.local}"
    )


def demo_offline_local() -> DemoResult:
    """Demo 3: network down + local model available → local selected."""
    router = build_router(
        models=[build_local_coding_model()],
        providers=[("ollama", StubLLMClient("ollama"))],
        network_checker=AlwaysOfflineNetworkChecker(),
    )
    selection = asyncio_run(
        router.select(
            RoutingRequest(capabilities={ModelCapability.CODING}, privacy=PrivacyLevel.PUBLIC)
        )
    )
    ok = selection.local and selection.provider == "ollama"
    return DemoResult(
        "demo_offline_local", ok, f"selected={selection.display} local={selection.local}"
    )


def demo_offline_no_model() -> DemoResult:
    """Demo 4: network down + no local model → ModelUnavailableError."""
    router = build_router(
        models=[build_cloud_coding_model()],
        providers=[("openai_compatible", StubLLMClient("openai_compatible"))],
        network_checker=AlwaysOfflineNetworkChecker(),
    )
    try:
        asyncio_run(
            router.select(
                RoutingRequest(capabilities={ModelCapability.CODING}, privacy=PrivacyLevel.PUBLIC)
            )
        )
        return DemoResult("demo_offline_no_model", False, "expected ModelUnavailableError")
    except ModelUnavailableError as exc:
        return DemoResult("demo_offline_no_model", True, f"got ModelUnavailableError: {exc}")


def demo_private_task() -> DemoResult:
    """Demo 5: PRIVATE task + local + cloud available → local selected."""
    router = build_router(
        models=[build_local_coding_model(), build_cloud_coding_model()],
        providers=[
            ("ollama", StubLLMClient("ollama")),
            ("openai_compatible", StubLLMClient("openai_compatible")),
        ],
        network_checker=AlwaysOfflineNetworkChecker(),
    )
    selection = asyncio_run(
        router.select(
            RoutingRequest(capabilities={ModelCapability.CODING}, privacy=PrivacyLevel.PRIVATE)
        )
    )
    ok = selection.local and selection.provider == "ollama"
    return DemoResult(
        "demo_private_task", ok, f"selected={selection.display} local={selection.local}"
    )


def demo_llm_planner() -> DemoResult:
    """Demo 6: LLM planner produces a validated structured plan."""
    from app.agents.interface import AgentCapability, AgentInfo, AgentInterface
    from app.llm.planner import LLMPlanner
    from app.runtime.loop.loop_context import LoopContext
    from app.runtime.loop.loop_policy import LoopPolicy
    from app.runtime.loop.loop_state import LoopState
    from app.runtime.models import Task
    from app.tools.registry import ToolRegistry

    class MathAgent(AgentInterface):
        @property
        def info(self) -> AgentInfo:
            return AgentInfo(
                agent_id="math.agent",
                name="Math",
                capabilities={AgentCapability.MATH},
            )

        async def execute(self, context):  # type: ignore[override]
            from app.agents.interface import AgentResult, AgentStatus

            return AgentResult(
                task_id=context.task_id,
                agent_id="math.agent",
                status=AgentStatus.COMPLETED,
                output=100,
            )

    plan_payload = {
        "goal": "Calculate 25 * 4",
        "steps": [
            {
                "description": "Compute 25 * 4 using the math agent.",
                "capability": "math",
                "agent_id": "math.agent",
                "expected_output": "100",
                "verification_requirements": ["result == 100"],
            }
        ],
        "assumptions": ["integer arithmetic"],
        "success_criteria": ["result == 100"],
    }
    client = StubLLMClient("ollama", structured_payload=plan_payload)
    providers = ProviderRegistry()
    providers.register("ollama", client)
    models = ModelRegistry()
    models.register(build_local_coding_model())
    router = ModelRouter(
        model_registry=models,
        provider_registry=providers,
        settings=_base_llm_settings(),
    )
    planner = LLMPlanner(router=router, settings=_base_llm_settings(), client_override=client)

    agent_registry = type("AR", (), {"exists": lambda self, aid: aid == "math.agent"})()
    tool_registry = ToolRegistry()
    state = LoopState(task_id="t1", goal="Calculate 25 * 4", success_criteria=["expected:100"])
    task = Task(
        task_id="t1", agent_id="math.agent", metadata={"discovery": {"agent_id": "math.agent"}}
    )
    context = LoopContext(
        state=state,
        task=task,
        agent_registry=agent_registry,  # type: ignore[arg-type]
        tool_registry=tool_registry,
        event_bus=InMemoryEventBus(),
        policy=LoopPolicy(),
    )
    result = asyncio_run(planner.plan(context))
    meta = result.reasoning_metadata
    ok = (
        meta.get("planner") == "llm"
        and result.next_action.target == "math.agent"
        and bool(meta.get("success_criteria"))
    )
    return DemoResult(
        "demo_llm_planner", ok, f"planner={meta.get('planner')} target={result.next_action.target}"
    )


def run_all_demos() -> list[DemoResult]:
    return [
        demo_local_selection(),
        demo_cloud_fallback(),
        demo_offline_local(),
        demo_offline_no_model(),
        demo_private_task(),
        demo_llm_planner(),
    ]


def asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)
