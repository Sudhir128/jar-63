"""Phase 5 demonstration: real local LLM runtime integration.

This module demonstrates the complete Phase 5 chain:

    Ollama health check → model discovery → capability sync →
    routing → LLM planning → tool execution → verification.

It can run against a **real** Ollama instance (if reachable) or fall back to
a **mocked** transport (for environments without Ollama). The demo never
fakes results — when mocked, it uses a deterministic httpx.MockTransport
that emulates Ollama's HTTP API.

Run it::

    python -m app.llm.phase5_demos            # auto-detect Ollama
    python -m app.llm.phase5_demos --mock     # force mocked transport
    python -m app.llm.phase5_demos --real     # force real Ollama
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.agents.registry import AgentRegistry
from app.config import LLMSettings
from app.events import InMemoryEventBus
from app.llm.health import LLMHealthChecker
from app.llm.models import (
    ModelCapability,
    ModelDefinition,
)
from app.llm.planner import LLMPlanner
from app.llm.providers.ollama import OLLAMA_PROVIDER_ID, OllamaClient
from app.llm.registry import ModelRegistry, ProviderRegistry
from app.llm.router import ModelRouter, RoutingRequest
from app.runtime.loop.service import LoopService
from app.runtime.loop.verification import CallableVerifier
from app.runtime.models import Task
from app.runtime.task_manager import TaskManager
from app.tools.impl import DEFAULT_TOOLS
from app.tools.policy import AllowAllToolPolicy
from app.tools.registry import ToolRegistry

__all__ = [
    "Phase5DemoResult",
    "run_phase5_demo",
    "run_calculator_e2e_demo",
    "run_health_check_demo",
    "run_routing_demo",
    "main",
]


@dataclass
class Phase5DemoResult:
    """Result of a Phase 5 demo run."""

    demo_name: str
    success: bool
    ollama_real: bool
    model: str
    health_status: str
    capabilities: list[str] = field(default_factory=list)
    final_status: str = ""
    final_response: Any = None
    error: str | None = None

    def summary(self) -> str:
        lines = [
            f"  demo:       {self.demo_name}",
            f"  ollama:     {'real' if self.ollama_real else 'mocked'}",
            f"  model:      {self.model}",
            f"  health:     {self.health_status}",
        ]
        if self.capabilities:
            lines.append(f"  capabilities: {', '.join(self.capabilities)}")
        if self.final_status:
            lines.append(f"  final:      {self.final_status}")
        if self.final_response is not None:
            lines.append(f"  response:   {self.final_response}")
        if self.error:
            lines.append(f"  error:      {self.error}")
        lines.append(f"  success:    {self.success}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Mock transport (deterministic Ollama emulation)
# ---------------------------------------------------------------------------

_CALC_PLAN = {
    "plan_id": "demo-plan",
    "goal": "Calculate 15 * 4",
    "steps": [
        {
            "step_id": "s1",
            "description": "Use the calculator to compute 15 * 4.",
            "capability": "arithmetic",
            "tool": "calculator",
            "tool_arguments": {"expression": "15 * 4"},
            "expected_output": "60",
            "verification_requirements": ["result == 60"],
        }
    ],
    "assumptions": [],
    "success_criteria": ["result == 60"],
}


class MockOllamaTransport:
    """Deterministic mock of Ollama's HTTP API for demos."""

    def __init__(self, *, chat_plan: dict[str, Any] | None = None) -> None:
        self.installed = ["qwen2.5-coder:7b"]
        self.chat_plan = chat_plan or _CALC_PLAN

    def handler(self, request: httpx.Request) -> httpx.Response:
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


def _make_settings() -> LLMSettings:
    return LLMSettings(
        _env_file=None,
        enabled=True,
        default_provider="ollama",
        default_model="qwen2.5-coder:7b",
        ollama_default_model="qwen2.5-coder:7b",
        routing_policy="local_first",
        request_timeout=30,
    )


async def _try_real_ollama() -> OllamaClient | None:
    """Probe for a real Ollama; return a client or None."""
    settings = _make_settings()
    client = OllamaClient(base_url=settings.ollama_base_url, timeout=settings.request_timeout)
    try:
        health = await client.check_health()
        if health.available:
            return client
    except Exception:  # noqa: BLE001
        pass
    await client.close()
    return None


def _make_mock_client(event_bus=None) -> OllamaClient:
    transport = MockOllamaTransport()
    http_client = httpx.AsyncClient(
        base_url="http://localhost:11434",
        transport=httpx.MockTransport(transport.handler),
        timeout=30.0,
    )
    return OllamaClient(
        base_url="http://localhost:11434",
        timeout=30.0,
        http_client=http_client,
        event_bus=event_bus,
    )


def _build_subsystems(
    client: OllamaClient, *, settings: LLMSettings, bus: InMemoryEventBus
) -> tuple[ModelRegistry, ProviderRegistry, LLMHealthChecker]:
    models = ModelRegistry()
    providers = ProviderRegistry()
    providers.register(OLLAMA_PROVIDER_ID, client)
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
        settings=settings,
        event_bus=bus,
    )
    return models, providers, checker


# ---------------------------------------------------------------------------
# Demo 1: Health check + discovery
# ---------------------------------------------------------------------------


async def run_health_check_demo(*, use_mock: bool | None = None) -> Phase5DemoResult:
    """Demo 1: Ollama health check + model discovery + capability sync."""
    settings = _make_settings()
    bus = InMemoryEventBus()
    real_client = None
    if use_mock is None:
        real_client = await _try_real_ollama()
        client = real_client if real_client else _make_mock_client(bus)
        is_real = real_client is not None
    elif use_mock:
        client = _make_mock_client(bus)
        is_real = False
    else:
        client = await _try_real_ollama()
        if client is None:
            return Phase5DemoResult(
                demo_name="health_check",
                success=False,
                ollama_real=False,
                model=settings.ollama_default_model,
                health_status="unavailable",
                error="Real Ollama not reachable and --mock not set.",
            )
        is_real = True

    models, providers, checker = _build_subsystems(client, settings=settings, bus=bus)
    snap = await checker.check()
    result = Phase5DemoResult(
        demo_name="health_check",
        success=snap.available,
        ollama_real=is_real,
        model=snap.model,
        health_status=snap.status,
        capabilities=snap.capabilities,
    )
    if not snap.available:
        result.error = snap.detail
    await client.close()
    return result


# ---------------------------------------------------------------------------
# Demo 2: Routing
# ---------------------------------------------------------------------------


async def run_routing_demo(*, use_mock: bool | None = None) -> Phase5DemoResult:
    """Demo 2: Model router selects the local model."""
    settings = _make_settings()
    bus = InMemoryEventBus()
    real_client = None
    if use_mock is None:
        real_client = await _try_real_ollama()
        client = real_client if real_client else _make_mock_client(bus)
        is_real = real_client is not None
    elif use_mock:
        client = _make_mock_client(bus)
        is_real = False
    else:
        client = await _try_real_ollama()
        if client is None:
            return Phase5DemoResult(
                demo_name="routing",
                success=False,
                ollama_real=False,
                model=settings.ollama_default_model,
                health_status="unavailable",
                error="Real Ollama not reachable.",
            )
        is_real = True

    models, providers, checker = _build_subsystems(client, settings=settings, bus=bus)
    await checker.check()
    router = ModelRouter(model_registry=models, provider_registry=providers, settings=settings)
    selection = await router.select(
        RoutingRequest(capabilities={ModelCapability.CHAT, ModelCapability.CODING})
    )
    result = Phase5DemoResult(
        demo_name="routing",
        success=selection.local,
        ollama_real=is_real,
        model=selection.model_id,
        health_status=checker.snapshot.status,
        final_status=f"selected {selection.provider}/{selection.model_id} (local={selection.local})",
    )
    await client.close()
    return result


# ---------------------------------------------------------------------------
# Demo 3: Full e2e calculator loop
# ---------------------------------------------------------------------------


async def run_calculator_e2e_demo(*, use_mock: bool | None = None) -> Phase5DemoResult:
    """Demo 3: Full chain — health → discovery → plan → tool → verify."""
    settings = _make_settings()
    bus = InMemoryEventBus()
    real_client = None
    if use_mock is None:
        real_client = await _try_real_ollama()
        client = real_client if real_client else _make_mock_client(bus)
        is_real = real_client is not None
    elif use_mock:
        client = _make_mock_client(bus)
        is_real = False
    else:
        client = await _try_real_ollama()
        if client is None:
            return Phase5DemoResult(
                demo_name="calculator_e2e",
                success=False,
                ollama_real=False,
                model=settings.ollama_default_model,
                health_status="unavailable",
                error="Real Ollama not reachable.",
            )
        is_real = True

    models, providers, checker = _build_subsystems(client, settings=settings, bus=bus)
    await checker.check()
    tools = ToolRegistry()
    for tool in DEFAULT_TOOLS():
        with contextlib.suppress(Exception):
            await tools.register(tool)
    router = ModelRouter(model_registry=models, provider_registry=providers, settings=settings)
    planner = LLMPlanner(router=router, settings=settings, event_bus=bus)
    service = LoopService(
        task_manager=TaskManager(),
        agent_registry=AgentRegistry(),
        tool_registry=tools,
        event_bus=bus,
        tool_policy=AllowAllToolPolicy(),
        llm_planner=planner,
    )

    def _check_result(output: Any) -> bool:
        return isinstance(output, dict) and output.get("result") == 60

    task = Task(task_id="demo-calc", input="Calculate 15 * 4")
    loop_result = await service.run_task_loop(
        task,
        goal="Calculate 15 * 4",
        success_criteria=["expected:60"],
        max_iterations=3,
        expected_output=60,
        verifier=CallableVerifier(_check_result, check_name="calculator_result"),
    )
    result = Phase5DemoResult(
        demo_name="calculator_e2e",
        success=loop_result.final_status.value == "success",
        ollama_real=is_real,
        model=settings.ollama_default_model,
        health_status=checker.snapshot.status,
        capabilities=checker.snapshot.capabilities,
        final_status=loop_result.final_status.value,
        final_response=loop_result.final_response,
    )
    if loop_result.failure_reason:
        result.error = loop_result.failure_reason
    await client.close()
    return result


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def run_phase5_demo(*, use_mock: bool | None = None) -> list[Phase5DemoResult]:
    """Run all Phase 5 demos."""
    results: list[Phase5DemoResult] = []
    print("=" * 60)
    print("JAR-63 Phase 5 Demo: Real Local LLM Runtime Integration")
    print("=" * 60)
    for demo_fn in [run_health_check_demo, run_routing_demo, run_calculator_e2e_demo]:
        name = demo_fn.__name__.replace("run_", "").replace("_demo", "")
        print(f"\n[{name}] running...")
        try:
            r = await demo_fn(use_mock=use_mock)
        except Exception as exc:  # noqa: BLE001
            r = Phase5DemoResult(
                demo_name=name,
                success=False,
                ollama_real=False,
                model="qwen2.5-coder:7b",
                health_status="error",
                error=f"{type(exc).__name__}: {exc}",
            )
        print(r.summary())
        results.append(r)
    print("\n" + "=" * 60)
    ok = sum(1 for r in results if r.success)
    print(f"Phase 5 demos: {ok}/{len(results)} succeeded")
    print("=" * 60)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="JAR-63 Phase 5 demos")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--mock", action="store_true", help="Force mocked Ollama transport")
    group.add_argument(
        "--real", action="store_true", help="Force real Ollama (fail if unreachable)"
    )
    args = parser.parse_args()
    use_mock = True if args.mock else (False if args.real else None)
    asyncio.run(run_phase5_demo(use_mock=use_mock))


if __name__ == "__main__":
    main()
