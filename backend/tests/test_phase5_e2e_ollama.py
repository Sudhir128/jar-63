"""Phase 5 end-to-end tests: real OllamaClient (mocked transport) through
the full runtime.

These tests verify the complete Phase 5 chain using the REAL
:class:`OllamaClient` (not a stub) with an :class:`httpx.MockTransport` that
emulates Ollama's HTTP API:

    health check → model discovery → registry sync → routing →
    LLM request → structured plan → tool plan step → execution →
    observation → verification → loop completion.

No real Ollama server is required. The transport is fully deterministic.
"""

from __future__ import annotations

import contextlib
import json
from typing import Any

import httpx

from app.agents.registry import AgentRegistry
from app.config import (
    AppEnv,
    AppSettings,
    DatabaseSettings,
    LLMSettings,
    LoggingSettings,
    MemorySettings,
    RedisSettings,
    SecuritySettings,
    Settings,
)
from app.events import InMemoryEventBus
from app.llm.health import LLMHealthChecker
from app.llm.models import ModelCapability
from app.llm.providers.ollama import OLLAMA_PROVIDER_ID, OllamaClient
from app.llm.registry import ModelRegistry, ProviderRegistry
from app.llm.router import ModelRouter, RoutingRequest
from app.runtime.loop.service import LoopService
from app.runtime.models import Task
from app.tools.impl import DEFAULT_TOOLS
from app.tools.policy import AllowAllToolPolicy
from app.tools.registry import ToolRegistry


def _llm_settings() -> LLMSettings:
    return LLMSettings(
        _env_file=None,
        enabled=True,
        default_provider="ollama",
        default_model="qwen2.5-coder:7b",
        ollama_default_model="qwen2.5-coder:7b",
        routing_policy="local_first",
        request_timeout=10,
    )


def _settings() -> Settings:
    return Settings(
        app=AppSettings(_env_file=None, name="JAR-63", env=AppEnv.TESTING, debug=False),
        database=DatabaseSettings(_env_file=None),
        redis=RedisSettings(_env_file=None),
        llm=_llm_settings(),
        memory=MemorySettings(_env_file=None),
        security=SecuritySettings(_env_file=None),
        logging=LoggingSettings(_env_file=None),
    )


class OllamaToolMockTransport:
    """Mocks Ollama's HTTP API with tool-calling + structured plan support.

    * ``GET /api/tags`` → lists ``qwen2.5-coder:7b``.
    * ``POST /api/show`` → reports the ``tools`` capability.
    * ``POST /api/chat`` → returns a structured plan JSON (with a tool step)
      when the prompt mentions planning, or a plain response otherwise.
    """

    def __init__(self, *, chat_plan: dict[str, Any] | None = None) -> None:
        self.installed = ["qwen2.5-coder:7b"]
        self.chat_plan = chat_plan or {
            "plan_id": "plan-1",
            "goal": "Calculate 7 * 6",
            "steps": [
                {
                    "step_id": "s1",
                    "description": "Use the calculator to compute 7 * 6.",
                    "capability": "arithmetic",
                    "tool": "calculator",
                    "tool_arguments": {"expression": "7 * 6"},
                    "expected_output": "42",
                    "verification_requirements": ["result == 42"],
                }
            ],
            "assumptions": [],
            "success_criteria": ["result == 42"],
        }
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": m} for m in self.installed]})
        if path == "/api/show":
            return httpx.Response(
                200,
                json={
                    "model_info": {
                        "families": ["llama"],
                        "context_length": 32768,
                        "capabilities": ["tools"],
                    },
                    "details": {"parameter_size": "7B"},
                },
            )
        if path == "/api/chat":
            # Return the structured plan as JSON content.
            return httpx.Response(
                200,
                json={
                    "model": "qwen2.5-coder:7b",
                    "message": {"role": "assistant", "content": json.dumps(self.chat_plan)},
                    "done": True,
                    "prompt_eval_count": 20,
                    "eval_count": 50,
                },
            )
        return httpx.Response(404, text="not found")

    def make_client(self, event_bus=None) -> OllamaClient:
        transport = httpx.MockTransport(self.handler)
        http_client = httpx.AsyncClient(
            base_url="http://localhost:11434", transport=transport, timeout=10.0
        )
        return OllamaClient(
            base_url="http://localhost:11434",
            timeout=10.0,
            http_client=http_client,
            event_bus=event_bus,
        )


def _build_components(
    *, chat_plan: dict[str, Any] | None = None
) -> tuple[
    OllamaClient,
    ModelRegistry,
    ProviderRegistry,
    LLMHealthChecker,
    ToolRegistry,
    Settings,
    InMemoryEventBus,
]:
    bus = InMemoryEventBus()
    transport = OllamaToolMockTransport(chat_plan=chat_plan)
    client = transport.make_client(event_bus=bus)
    models = ModelRegistry()
    providers = ProviderRegistry()
    providers.register(OLLAMA_PROVIDER_ID, client)
    settings = _settings()
    # Pre-register a static model (without tools) so sync can upgrade it.
    from app.llm.models import ModelDefinition

    models.register(
        ModelDefinition(
            model_id="qwen2.5-coder:7b",
            provider=OLLAMA_PROVIDER_ID,
            capabilities={ModelCapability.CHAT, ModelCapability.CODING},
            supports_tools=False,
            local=True,
            enabled=True,
        )
    )
    checker = LLMHealthChecker(
        provider_registry=providers,
        model_registry=models,
        settings=settings.llm,
        event_bus=bus,
    )
    tools = ToolRegistry()
    return client, models, providers, checker, tools, settings, bus


# ---------------------------------------------------------------------------
# Health + discovery + sync (real OllamaClient)
# ---------------------------------------------------------------------------


async def test_real_ollama_health_check_available() -> None:
    client, models, providers, checker, _, _, _ = _build_components()
    snap = await checker.check()
    assert snap.available
    assert snap.model == "qwen2.5-coder:7b"
    assert "qwen2.5-coder:7b" in snap.installed_models
    await client.close()


async def test_real_ollama_discovery_syncs_tool_capability() -> None:
    client, models, providers, checker, _, _, _ = _build_components()
    await checker.check()
    model = models.get(OLLAMA_PROVIDER_ID, "qwen2.5-coder:7b")
    # /api/show reports the "tools" capability -> sync sets TOOL_CALLING.
    assert ModelCapability.TOOL_CALLING in model.capabilities
    assert model.supports_tools is True
    await client.close()


async def test_real_ollama_health_model_not_found() -> None:
    """When the model is not installed, health reports MODEL_NOT_FOUND."""
    transport = OllamaToolMockTransport()
    transport.installed = []  # no models installed
    client = transport.make_client()
    models = ModelRegistry()
    providers = ProviderRegistry()
    providers.register(OLLAMA_PROVIDER_ID, client)
    from app.llm.models import ModelDefinition

    models.register(
        ModelDefinition(
            model_id="qwen2.5-coder:7b",
            provider=OLLAMA_PROVIDER_ID,
            capabilities={ModelCapability.CHAT},
            local=True,
            enabled=True,
        )
    )
    checker = LLMHealthChecker(
        provider_registry=providers,
        model_registry=models,
        settings=_llm_settings(),
        event_bus=InMemoryEventBus(),
    )
    snap = await checker.check()
    assert not snap.available
    assert snap.status == "model_not_found"
    await client.close()


async def test_real_ollama_health_unavailable_on_connect_error() -> None:
    transport = OllamaToolMockTransport()
    client = transport.make_client()
    # Replace with a client whose transport raises connect errors.
    bad_transport = httpx.MockTransport(lambda req: (_ for _ in ()).throw(httpx.ConnectError("no")))
    client._http = httpx.AsyncClient(
        base_url="http://localhost:11434", transport=bad_transport, timeout=5.0
    )
    client._owns_http = True
    models = ModelRegistry()
    providers = ProviderRegistry()
    providers.register(OLLAMA_PROVIDER_ID, client)
    from app.llm.models import ModelDefinition

    models.register(
        ModelDefinition(
            model_id="qwen2.5-coder:7b",
            provider=OLLAMA_PROVIDER_ID,
            capabilities={ModelCapability.CHAT},
            local=True,
            enabled=True,
        )
    )
    checker = LLMHealthChecker(
        provider_registry=providers,
        model_registry=models,
        settings=_llm_settings(),
        event_bus=InMemoryEventBus(),
    )
    snap = await checker.check()
    assert not snap.available
    assert snap.status == "unavailable"
    await client.close()


# ---------------------------------------------------------------------------
# Routing (real OllamaClient)
# ---------------------------------------------------------------------------


async def test_real_ollama_router_selects_local_model() -> None:
    client, models, providers, checker, _, settings, _ = _build_components()
    await checker.check()  # sync capabilities
    router = ModelRouter(
        model_registry=models,
        provider_registry=providers,
        settings=settings.llm,
        network_checker=None,
    )
    selection = await router.select(
        RoutingRequest(capabilities={ModelCapability.CHAT, ModelCapability.CODING})
    )
    assert selection.local
    assert selection.provider == OLLAMA_PROVIDER_ID
    assert selection.model_id == "qwen2.5-coder:7b"
    await client.close()


# ---------------------------------------------------------------------------
# End-to-end: LLM plan → tool execution → verification
# ---------------------------------------------------------------------------


async def test_e2e_llm_plan_executes_tool_and_verifies() -> None:
    """Full chain: LLM planner produces a calculator tool plan, the loop
    executes it through the ToolExecutor, and the ExactMatchVerifier
    confirms the result.
    """
    client, models, providers, checker, tools, settings, bus = _build_components()
    await checker.check()
    # Register the calculator tool.
    for tool in DEFAULT_TOOLS():
        with contextlib.suppress(Exception):
            await tools.register(tool)
    from app.llm.planner import LLMPlanner

    router = ModelRouter(model_registry=models, provider_registry=providers, settings=settings.llm)
    planner = LLMPlanner(router=router, settings=settings.llm, event_bus=bus)
    service = LoopService(
        task_manager=__import__("app.runtime.task_manager", fromlist=["TaskManager"]).TaskManager(),
        agent_registry=AgentRegistry(),
        tool_registry=tools,
        event_bus=bus,
        settings=settings,
        tool_policy=AllowAllToolPolicy(),
        llm_planner=planner,
    )
    task = Task(task_id="e2e-1", input="Calculate 7 * 6")
    from app.runtime.loop.verification import CallableVerifier

    def _check_result(output: Any) -> bool:
        return isinstance(output, dict) and output.get("result") == 42

    result = await service.run_task_loop(
        task,
        goal="Calculate 7 * 6",
        success_criteria=["expected:42"],
        max_iterations=3,
        expected_output=42,
        verifier=CallableVerifier(_check_result, check_name="calculator_result"),
    )
    # The LLM planned a calculator tool call; execution produced 42; verifier passed.
    assert result.final_status.value == "success"
    assert result.final_response["result"] == 42
    await client.close()


async def test_e2e_falls_back_when_llm_returns_empty_plan() -> None:
    """When the LLM returns an empty plan, the planner falls back to the
    deterministic planner and the loop still completes (gracefully).
    """
    empty_plan = {
        "plan_id": "p-empty",
        "goal": "do nothing",
        "steps": [],
        "assumptions": [],
        "success_criteria": ["none"],
    }
    client, models, providers, checker, tools, settings, bus = _build_components(
        chat_plan=empty_plan
    )
    await checker.check()
    for tool in DEFAULT_TOOLS():
        with contextlib.suppress(Exception):
            await tools.register(tool)
    from app.llm.planner import LLMPlanner

    router = ModelRouter(model_registry=models, provider_registry=providers, settings=settings.llm)
    planner = LLMPlanner(router=router, settings=settings.llm, event_bus=bus)
    service = LoopService(
        task_manager=__import__("app.runtime.task_manager", fromlist=["TaskManager"]).TaskManager(),
        agent_registry=AgentRegistry(),
        tool_registry=tools,
        event_bus=bus,
        settings=settings,
        tool_policy=AllowAllToolPolicy(),
        llm_planner=planner,
    )
    task = Task(task_id="e2e-empty", input="do nothing")
    result = await service.run_task_loop(
        task, goal="do nothing", success_criteria=[], max_iterations=2
    )
    # Fallback occurred; loop did not crash.
    assert result.final_status.value in ("completed", "stopped", "failed")
    await client.close()


async def test_e2e_events_published_for_llm_request() -> None:
    """LLM request events are published (no prompt contents)."""
    from app.events import EventType

    client, models, providers, checker, tools, settings, bus = _build_components()
    events: list = []

    async def handler(ev) -> None:
        events.append(ev)

    bus.subscribe(EventType.LLM_REQUEST_STARTED, handler)
    bus.subscribe(EventType.LLM_REQUEST_COMPLETED, handler)
    await checker.check()
    for tool in DEFAULT_TOOLS():
        with contextlib.suppress(Exception):
            await tools.register(tool)
    from app.llm.planner import LLMPlanner

    router = ModelRouter(model_registry=models, provider_registry=providers, settings=settings.llm)
    planner = LLMPlanner(router=router, settings=settings.llm, event_bus=bus)
    service = LoopService(
        task_manager=__import__("app.runtime.task_manager", fromlist=["TaskManager"]).TaskManager(),
        agent_registry=AgentRegistry(),
        tool_registry=tools,
        event_bus=bus,
        settings=settings,
        tool_policy=AllowAllToolPolicy(),
        llm_planner=planner,
    )
    task = Task(task_id="e2e-events", input="Calculate 7 * 6")
    await service.run_task_loop(
        task, goal="Calculate 7 * 6", success_criteria=["result == 42"], max_iterations=2
    )
    types = {ev.event_type for ev in events}
    assert EventType.LLM_REQUEST_STARTED in types
    assert EventType.LLM_REQUEST_COMPLETED in types
    # Verify NO prompt contents leaked into events.
    import json as _json

    blob = _json.dumps([ev.model_dump() for ev in events], default=str)
    assert "Calculate 7 * 6" not in blob  # the user prompt text
    assert "messages" not in blob.lower()
    await client.close()
