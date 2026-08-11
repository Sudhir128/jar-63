"""Optional integration tests against a real Ollama instance.

These are NOT part of the normal unit test suite. Run them explicitly:

    pytest -m ollama

They are skipped automatically unless ``OLLAMA_INTEGRATION=1`` is set AND a
real Ollama server is reachable at ``OLLAMA_BASE_URL`` with the configured
model installed. They never download or modify models.
"""

from __future__ import annotations

import os

import pytest

ollama_integration = pytest.mark.ollama

_SKIP_REASON = (
    "Set OLLAMA_INTEGRATION=1 and ensure a real Ollama server is reachable "
    "with the configured model installed to run this integration test."
)


def _integration_enabled() -> bool:
    return os.environ.get("OLLAMA_INTEGRATION", "").lower() in ("1", "true", "yes")


pytestmark = [
    ollama_integration,
    pytest.mark.skipif(not _integration_enabled(), reason=_SKIP_REASON),
]


@pytest.fixture()
def real_ollama_client():
    from app.config import get_settings
    from app.llm.providers.ollama import OllamaClient

    settings = get_settings().llm
    return OllamaClient(base_url=settings.ollama_base_url, timeout=settings.request_timeout)


async def test_ollama_reachable(real_ollama_client) -> None:
    health = await real_ollama_client.check_health()
    assert health.status == "available", f"Ollama not reachable: {health.detail}"
    await real_ollama_client.close()


async def test_ollama_model_available(real_ollama_client) -> None:
    from app.config import get_settings

    model = get_settings().llm.ollama_default_model
    health = await real_ollama_client.check_health(model)
    assert health.status == "available", (
        f"Model '{model}' not available. Install it with: ollama pull {model}"
    )
    await real_ollama_client.close()


async def test_ollama_simple_request(real_ollama_client) -> None:
    from app.config import get_settings
    from app.llm.models import LLMMessage, LLMRequest, MessageRole

    model = get_settings().llm.ollama_default_model
    resp = await real_ollama_client.generate(
        LLMRequest(
            model=model,
            messages=[
                LLMMessage(role=MessageRole.USER, content="Reply with the single word: pong")
            ],
            temperature=0.0,
        )
    )
    assert resp.content
    assert resp.provider == "ollama"
    await real_ollama_client.close()


async def test_ollama_structured_request(real_ollama_client) -> None:
    from pydantic import BaseModel as PydanticBaseModel

    from app.config import get_settings
    from app.llm.models import LLMMessage, LLMRequest, MessageRole, StructuredOutputSpec

    class Answer(PydanticBaseModel):
        value: int

    model = get_settings().llm.ollama_default_model
    spec = StructuredOutputSpec(name="answer", response_model=Answer)
    resp = await real_ollama_client.generate_structured(
        LLMRequest(
            model=model,
            messages=[
                LLMMessage(
                    role=MessageRole.USER, content="What is 6 times 7? Reply as JSON {value: int}."
                )
            ],
            temperature=0.0,
            structured_output=spec,
        )
    )
    parsed = resp.metadata.get("parsed")
    assert parsed is not None
    assert parsed["value"] == 42
    await real_ollama_client.close()


# ---------------------------------------------------------------------------
# Phase 5: Health checker + discovery + full loop e2e
# ---------------------------------------------------------------------------


@pytest.fixture()
def llm_components(real_ollama_client):
    """Build the LLM subsystem against a real Ollama instance."""
    from app.config import get_settings
    from app.events import InMemoryEventBus
    from app.llm.health import LLMHealthChecker
    from app.llm.models import ModelCapability, ModelDefinition
    from app.llm.providers.ollama import OLLAMA_PROVIDER_ID
    from app.llm.registry import ModelRegistry, ProviderRegistry

    settings = get_settings().llm
    models = ModelRegistry()
    providers = ProviderRegistry()
    providers.register(OLLAMA_PROVIDER_ID, real_ollama_client)
    # Pre-register a static model (without tools) so sync can upgrade it.
    models.register(
        ModelDefinition(
            model_id=settings.ollama_default_model,
            provider=OLLAMA_PROVIDER_ID,
            capabilities={ModelCapability.CHAT, ModelCapability.CODING},
            supports_tools=False,
            local=True,
            enabled=True,
        )
    )
    bus = InMemoryEventBus()
    checker = LLMHealthChecker(
        provider_registry=providers,
        model_registry=models,
        settings=settings,
        event_bus=bus,
    )
    return {
        "client": real_ollama_client,
        "models": models,
        "providers": providers,
        "checker": checker,
        "settings": settings,
        "bus": bus,
    }


async def test_ollama_health_checker_reports_available(llm_components) -> None:
    snap = await llm_components["checker"].check()
    assert snap.available, f"Health check failed: {snap.detail}"
    assert snap.model == llm_components["settings"].ollama_default_model
    assert snap.model_available is True
    assert llm_components["settings"].ollama_default_model in snap.installed_models


async def test_ollama_discovery_syncs_capabilities(llm_components) -> None:
    """Discovery via /api/show syncs real capabilities into the registry."""
    from app.llm.models import ModelCapability

    await llm_components["checker"].check()
    model = llm_components["models"].get("ollama", llm_components["settings"].ollama_default_model)
    # qwen2.5-coder supports chat at minimum.
    assert ModelCapability.CHAT in model.capabilities


async def test_ollama_router_selects_local_model(llm_components) -> None:
    from app.llm.models import ModelCapability
    from app.llm.router import ModelRouter, RoutingRequest

    await llm_components["checker"].check()
    router = ModelRouter(
        model_registry=llm_components["models"],
        provider_registry=llm_components["providers"],
        settings=llm_components["settings"],
    )
    selection = await router.select(
        RoutingRequest(capabilities={ModelCapability.CHAT, ModelCapability.CODING})
    )
    assert selection.local
    assert selection.provider == "ollama"


async def test_ollama_e2e_loop_with_calculator(llm_components) -> None:
    """Full Phase 5 chain against real Ollama: plan → tool → verify.

    This is the end-to-end integration test. It requires qwen2.5-coder:7b to
    produce a valid structured plan that calls the calculator tool. If the
    model's output doesn't validate, the planner falls back to the
    deterministic planner (which also plans a calculator call), so the loop
    should still succeed.
    """
    from app.agents.registry import AgentRegistry
    from app.llm.planner import LLMPlanner
    from app.llm.router import ModelRouter
    from app.runtime.loop.service import LoopService
    from app.runtime.loop.verification import CallableVerifier
    from app.runtime.models import Task
    from app.runtime.task_manager import TaskManager
    from app.tools.impl import DEFAULT_TOOLS
    from app.tools.policy import AllowAllToolPolicy

    await llm_components["checker"].check()
    from app.tools.registry import ToolRegistry

    tools = ToolRegistry()
    import contextlib

    for tool in DEFAULT_TOOLS():
        with contextlib.suppress(Exception):
            await tools.register(tool)
    router = ModelRouter(
        model_registry=llm_components["models"],
        provider_registry=llm_components["providers"],
        settings=llm_components["settings"],
    )
    planner = LLMPlanner(
        router=router, settings=llm_components["settings"], event_bus=llm_components["bus"]
    )
    service = LoopService(
        task_manager=TaskManager(),
        agent_registry=AgentRegistry(),
        tool_registry=tools,
        event_bus=llm_components["bus"],
        tool_policy=AllowAllToolPolicy(),
        llm_planner=planner,
    )

    def _check_result(output):
        return isinstance(output, dict) and output.get("result") == 42

    task = Task(task_id="int-e2e", input="Calculate 7 * 6")
    result = await service.run_task_loop(
        task,
        goal="Calculate 7 * 6",
        success_criteria=["expected:42"],
        max_iterations=3,
        expected_output=42,
        verifier=CallableVerifier(_check_result, check_name="calculator_result"),
    )
    # The loop should succeed (either via LLM plan or deterministic fallback).
    assert result.final_status.value == "success", (
        f"Loop did not succeed: {result.final_status.value} — {result.failure_reason}"
    )
    assert result.final_response["result"] == 42
