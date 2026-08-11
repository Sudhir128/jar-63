"""Shared LLM test helpers and fixtures.

Tests run fully offline. The Ollama HTTP API is mocked via
:class:`httpx.MockTransport`; no real Ollama server or cloud provider is
required.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from app.config import LLMSettings
from app.llm.client import HealthStatus, LLMClient, ProviderHealth
from app.llm.models import LLMRequest, LLMResponse, ModelDefinition
from app.llm.providers.ollama import OLLAMA_PROVIDER_ID, OllamaClient
from app.llm.providers.openai_compatible import (
    OPENAI_COMPATIBLE_PROVIDER_ID,
    OpenAICompatibleClient,
)


def make_llm_settings(
    *,
    enabled: bool = True,
    allow_cloud_private: bool = False,
    openai_compat_enabled: bool = False,
) -> LLMSettings:
    return LLMSettings(
        _env_file=None,
        enabled=enabled,
        default_provider="ollama",
        default_model="qwen2.5-coder:7b",
        routing_policy="local_first",
        request_timeout=10,
        verbose_logging=False,
        allow_cloud_for_private=allow_cloud_private,
        ollama_base_url="http://localhost:11434",
        ollama_default_model="qwen2.5-coder:7b",
        openai_compatible_enabled=openai_compat_enabled,
        openai_compatible_base_url="http://cloud.example.com" if openai_compat_enabled else "",
        openai_compatible_model="cloud-coder:32b" if openai_compat_enabled else "",
    )


class OllamaMockTransport:
    """A configurable httpx.MockTransport that emulates the Ollama HTTP API."""

    def __init__(
        self,
        *,
        installed_models: list[str] | None = None,
        chat_response: dict[str, Any] | None = None,
        chat_status: int = 200,
        tags_status: int = 200,
        raise_connect: bool = False,
        raise_timeout: bool = False,
    ) -> None:
        self.installed_models = (
            installed_models if installed_models is not None else ["qwen2.5-coder:7b"]
        )
        self.chat_response = chat_response or {
            "model": "qwen2.5-coder:7b",
            "message": {"role": "assistant", "content": "Hello!"},
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 3,
        }
        self.chat_status = chat_status
        self.tags_status = tags_status
        self.raise_connect = raise_connect
        self.raise_timeout = raise_timeout
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.raise_connect:
            raise httpx.ConnectError("mock connect error")
        if self.raise_timeout:
            raise httpx.TimeoutException("mock timeout")

        path = request.url.path
        if path == "/api/tags":
            if self.tags_status != 200:
                return httpx.Response(self.tags_status, text="error")
            return httpx.Response(
                200,
                json={"models": [{"name": m} for m in self.installed_models]},
            )
        if path == "/api/chat":
            if self.chat_status == 404:
                return httpx.Response(404, json={"error": "model not found"})
            if self.chat_status != 200:
                return httpx.Response(self.chat_status, text="error")
            return httpx.Response(200, json=self.chat_response)
        if path == "/api/show":
            return httpx.Response(
                200,
                json={
                    "model_info": {"families": ["llama"], "context_length": 32768},
                    "details": {"parameter_size": "7B"},
                },
            )
        return httpx.Response(404, text="not found")

    def make_client(self, **kwargs: Any) -> OllamaClient:
        transport = httpx.MockTransport(self.handler)
        http_client = httpx.AsyncClient(
            base_url="http://localhost:11434", transport=transport, timeout=10.0
        )
        return OllamaClient(
            base_url="http://localhost:11434",
            timeout=10.0,
            http_client=http_client,
            event_bus=kwargs.get("event_bus"),
        )


class OpenAICompatMockTransport:
    """A configurable httpx.MockTransport for an OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        completion_response: dict[str, Any] | None = None,
        status: int = 200,
        models_status: int = 200,
        raise_connect: bool = False,
    ) -> None:
        self.completion_response = completion_response or {
            "id": "chatcmpl-1",
            "model": "cloud-coder:32b",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hi from cloud"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
        }
        self.status = status
        self.models_status = models_status
        self.raise_connect = raise_connect
        self.requests: list[httpx.Request] = []
        self.auth_headers: list[str | None] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.auth_headers.append(request.headers.get("authorization"))
        if self.raise_connect:
            raise httpx.ConnectError("mock connect error")
        if request.url.path.endswith("/v1/models"):
            if self.models_status != 200:
                return httpx.Response(self.models_status, text="error")
            return httpx.Response(200, json={"data": [{"id": "cloud-coder:32b"}]})
        if request.url.path.endswith("/v1/chat/completions"):
            if self.status != 200:
                return httpx.Response(self.status, text="error")
            return httpx.Response(200, json=self.completion_response)
        return httpx.Response(404, text="not found")

    def make_client(self, *, api_key: str | None = None, **kwargs: Any) -> OpenAICompatibleClient:
        transport = httpx.MockTransport(self.handler)
        http_client = httpx.AsyncClient(transport=transport, timeout=10.0)
        return OpenAICompatibleClient(
            base_url="http://cloud.example.com",
            model="cloud-coder:32b",
            api_key=api_key,
            timeout=10.0,
            http_client=http_client,
            event_bus=kwargs.get("event_bus"),
        )


class StubLLMClient(LLMClient):
    """In-memory LLM client returning canned responses (no network)."""

    def __init__(
        self,
        provider_name: str,
        *,
        content: str = "",
        structured_payload: dict[str, Any] | None = None,
        health: ProviderHealth | None = None,
        raise_error: Exception | None = None,
    ) -> None:
        self._provider = provider_name
        self._content = content
        self._structured = structured_payload
        self._health = health or ProviderHealth(
            status=HealthStatus.AVAILABLE, provider=provider_name
        )
        self._raise = raise_error
        self.generate_calls: list[LLMRequest] = []

    @property
    def provider_name(self) -> str:
        return self._provider

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.generate_calls.append(request)
        if self._raise:
            raise self._raise
        return LLMResponse(content=self._content, model=request.model, provider=self._provider)

    async def generate_structured(
        self, request: LLMRequest, spec: Any | None = None
    ) -> LLMResponse:
        self.generate_calls.append(request)
        if self._raise:
            raise self._raise
        content = json.dumps(self._structured) if self._structured else "{}"
        return LLMResponse(
            content=content,
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


def local_coding_model(model_id: str = "qwen2.5-coder:7b") -> ModelDefinition:
    from app.llm.models import ModelCapability

    return ModelDefinition(
        model_id=model_id,
        provider=OLLAMA_PROVIDER_ID,
        display_name=model_id,
        capabilities={ModelCapability.CHAT, ModelCapability.CODING, ModelCapability.REASONING},
        context_window=32768,
        supports_structured_output=True,
        coding_score=0.8,
        local=True,
        enabled=True,
    )


def cloud_coding_model(model_id: str = "cloud-coder:32b") -> ModelDefinition:
    from app.llm.models import ModelCapability

    return ModelDefinition(
        model_id=model_id,
        provider=OPENAI_COMPATIBLE_PROVIDER_ID,
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


@pytest.fixture()
def _llm_helpers_marker():
    """Marker so pytest collects this module (fixtures moved to conftest.py)."""
    return None
