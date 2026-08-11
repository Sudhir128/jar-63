"""Generic OpenAI-compatible LLM provider.

This is NOT a requirement to use OpenAI. It exists for compatibility with
open/open-weight model hosting services and other servers that expose an
OpenAI-compatible ``/v1/chat/completions`` API.

Configuration (all optional; if disabled, JAR-63 works with Ollama only):

* ``OPENAI_COMPATIBLE_BASE_URL``
* ``OPENAI_COMPATIBLE_API_KEY`` (optional — never hardcoded)
* ``OPENAI_COMPATIBLE_MODEL``

The API key is sent only as a ``Bearer`` Authorization header and is never
logged. If no compatible provider is configured, this client is not created.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from app.core.identifiers import generate_id
from app.core.logging import get_logger
from app.events import Event, EventBus, EventType
from app.llm.client import HealthStatus, LLMClient, ProviderHealth
from app.llm.errors import (
    InvalidLLMResponseError,
    InvalidStructuredOutputError,
    LLMError,
)
from app.llm.models import (
    FinishReason,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMToolCall,
    LLMToolDefinition,
    LLMUsage,
    StructuredOutputSpec,
)
from app.llm.providers.base import BaseProviderMixin, normalize_timeout

logger = get_logger("llm.openai_compatible")

__all__ = ["OpenAICompatibleClient", "OPENAI_COMPATIBLE_PROVIDER_ID"]

OPENAI_COMPATIBLE_PROVIDER_ID = "openai_compatible"


class OpenAICompatibleClient(BaseProviderMixin, LLMClient):
    """LLM client for OpenAI-compatible HTTP endpoints."""

    provider_name = OPENAI_COMPATIBLE_PROVIDER_ID

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 60.0,
        verbose_logging: bool = False,
        http_client: httpx.AsyncClient | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("OpenAI-compatible base_url is required")
        if not model:
            raise ValueError("OpenAI-compatible model is required")
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._default_timeout = timeout
        self._verbose = verbose_logging
        self._event_bus = event_bus
        self._http = http_client
        self._owns_http = http_client is None

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            # Authorization header is never logged (see core/logging masking).
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self._default_timeout)
        return self._http

    async def generate(self, request: LLMRequest) -> LLMResponse:
        return await self._generate(request, structured=False)

    async def generate_structured(
        self, request: LLMRequest, spec: Any | None = None
    ) -> LLMResponse:
        spec = spec or request.structured_output
        if spec is None:
            raise InvalidStructuredOutputError("No structured output spec provided")
        return await self._generate(request, structured=True, spec=spec)

    async def check_health(self, model: str | None = None) -> ProviderHealth:
        # The canonical liveness check is a tiny models list request.
        try:
            client = await self._client()
            resp = await client.get(
                f"{self._base_url}/v1/models",
                headers=self._headers(),
                timeout=min(self._default_timeout, 10.0),
            )
        except httpx.TimeoutException:
            return ProviderHealth(status=HealthStatus.TIMEOUT, provider=self.provider_name)
        except httpx.ConnectError:
            return ProviderHealth(status=HealthStatus.UNAVAILABLE, provider=self.provider_name)
        except Exception as exc:  # noqa: BLE001
            return ProviderHealth(
                status=HealthStatus.INVALID_RESPONSE,
                provider=self.provider_name,
                detail=type(exc).__name__,
            )
        if resp.status_code in (401, 403):
            return ProviderHealth(
                status=HealthStatus.UNAVAILABLE,
                provider=self.provider_name,
                detail="authentication failed",
            )
        if resp.status_code != 200:
            return ProviderHealth(
                status=HealthStatus.UNAVAILABLE,
                provider=self.provider_name,
                detail=f"status {resp.status_code}",
            )
        return ProviderHealth(
            status=HealthStatus.AVAILABLE, provider=self.provider_name, model=model or self._model
        )

    async def get_model_info(self, model: str) -> Any:
        # Cloud model info is not assumed; return None (no local install concept).
        return None

    async def close(self) -> None:
        if self._http is not None and self._owns_http:
            await self._http.aclose()
            self._http = None

    # --- internals ---

    async def _generate(
        self,
        request: LLMRequest,
        *,
        structured: bool,
        spec: StructuredOutputSpec | None = None,
    ) -> LLMResponse:
        timeout = normalize_timeout(request, self._default_timeout)
        self._log_request(request, verbose=self._verbose)
        await self._publish_started(request)

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [_msg_to_dict(m) for m in request.messages],
            "stream": False,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if structured and spec is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": spec.name, "schema": spec.to_json_schema()},
            }
        if request.tools:
            payload["tools"] = [_tool_def_to_openai(t) for t in request.tools]

        started = time.monotonic()
        try:
            client = await self._client()
            resp = await client.post(
                f"{self._base_url}/v1/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=timeout,
            )
        except Exception as exc:
            err = self._classify_transport_error(exc, provider=self.provider_name)
            await self._publish_failed(request, err)
            raise err from exc

        latency_ms = int((time.monotonic() - started) * 1000)

        if resp.status_code != 200:
            err = self._classify_http_status(resp.status_code, provider=self.provider_name)
            await self._publish_failed(request, err)
            raise err

        try:
            data = resp.json()
        except Exception as exc:
            err = InvalidLLMResponseError(f"{self.provider_name}: invalid JSON: {exc}")
            await self._publish_failed(request, err)
            raise err from exc

        response = _parse_completion(data, request.model, self.provider_name)
        if structured and spec is not None:
            response = _attach_parsed(response, spec)
        self._log_response(
            request.model,
            self.provider_name,
            latency_ms=latency_ms,
            success=True,
            verbose=self._verbose,
            content=response.content,
        )
        await self._publish_completed(request, response, latency_ms)
        return response

    async def _publish_started(self, request: LLMRequest) -> None:
        if self._event_bus is None:
            return
        await self._event_bus.publish(
            Event.create(
                EventType.LLM_REQUEST_STARTED,
                payload={
                    "provider": self.provider_name,
                    "model": request.model,
                    "request_id": request.request_id,
                    "privacy": request.privacy.value,
                    "structured": request.structured_output is not None,
                },
                metadata=request.metadata,
            )
        )

    async def _publish_completed(
        self, request: LLMRequest, response: LLMResponse, latency_ms: int
    ) -> None:
        if self._event_bus is None:
            return
        await self._event_bus.publish(
            Event.create(
                EventType.LLM_REQUEST_COMPLETED,
                payload={
                    "provider": self.provider_name,
                    "model": request.model,
                    "request_id": request.request_id,
                    "latency_ms": latency_ms,
                    "usage": response.usage.model_dump(),
                    "finish_reason": response.finish_reason.value,
                },
                metadata=request.metadata,
            )
        )

    async def _publish_failed(self, request: LLMRequest, error: LLMError) -> None:
        logger.bind(
            provider=self.provider_name, model=request.model, error=type(error).__name__
        ).warning("LLM request failed: {}", str(error))
        if self._event_bus is None:
            return
        await self._event_bus.publish(
            Event.create(
                EventType.LLM_REQUEST_FAILED,
                payload={
                    "provider": self.provider_name,
                    "model": request.model,
                    "request_id": request.request_id,
                    "error": type(error).__name__,
                    "detail": str(error)[:300],
                },
                metadata=request.metadata,
            )
        )


def _msg_to_dict(m: LLMMessage) -> dict[str, Any]:
    """Convert an :class:`LLMMessage` to the OpenAI-compatible chat format."""
    d: dict[str, Any] = {"role": m.role.value, "content": m.content}
    if m.name:
        d["name"] = m.name
    if m.tool_calls:
        d["tool_calls"] = [_tool_call_to_openai(tc) for tc in m.tool_calls]
    if m.tool_call_id:
        d["tool_call_id"] = m.tool_call_id
    return d


def _tool_def_to_openai(defn: LLMToolDefinition) -> dict[str, Any]:
    """Map an :class:`LLMToolDefinition` to the OpenAI tool format."""
    return {
        "type": "function",
        "function": {
            "name": defn.name,
            "description": defn.description,
            "parameters": defn.parameters or {"type": "object", "properties": {}},
        },
    }


def _tool_call_to_openai(tc: LLMToolCall) -> dict[str, Any]:
    """Map an :class:`LLMToolCall` to the OpenAI assistant-message tool_call format."""
    return {
        "id": tc.id,
        "type": "function",
        "function": {
            "name": tc.name,
            "arguments": json.dumps(dict(tc.arguments), default=str),
        },
    }


def _parse_tool_calls(raw: list[dict[str, Any]]) -> list[LLMToolCall]:
    """Parse OpenAI-style ``message.tool_calls`` into typed :class:`LLMToolCall`.

    OpenAI returns ``arguments`` as a JSON string. Malformed entries are
    skipped — they never reach the :class:`ToolExecutor`.
    """
    calls: list[LLMToolCall] = []
    for raw_tc in raw or []:
        func = raw_tc.get("function") or {}
        name = func.get("name")
        if not name:
            continue
        raw_args = func.get("arguments")
        if raw_args is None:
            args = {}
        elif isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except (json.JSONDecodeError, TypeError):
                args = {}
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            args = {}
        if not isinstance(args, dict):
            args = {}
        call_id = raw_tc.get("id") or generate_id("call")
        calls.append(LLMToolCall(id=call_id, name=name, arguments=args))
    return calls


_FINISH_MAP = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "tool_calls": FinishReason.TOOL_CALLS,
    "content_filter": FinishReason.CONTENT_FILTER,
}


def _parse_completion(data: dict[str, Any], model: str, provider: str) -> LLMResponse:
    choices = data.get("choices") or []
    if not choices:
        raise InvalidLLMResponseError(f"{provider}: response has no choices")
    choice = choices[0]
    msg = choice.get("message") or {}
    content = msg.get("content", "") or ""
    tool_calls = _parse_tool_calls(msg.get("tool_calls") or [])
    finish = _FINISH_MAP.get(choice.get("finish_reason", "stop"), FinishReason.UNKNOWN)
    if tool_calls and finish is not FinishReason.TOOL_CALLS:
        finish = FinishReason.TOOL_CALLS
    usage_data = data.get("usage") or {}
    usage = LLMUsage(
        prompt_tokens=usage_data.get("prompt_tokens"),
        completion_tokens=usage_data.get("completion_tokens"),
        total_tokens=usage_data.get("total_tokens"),
    )
    return LLMResponse(
        content=content,
        model=data.get("model", model),
        provider=provider,
        usage=usage,
        finish_reason=finish,
        tool_calls=tool_calls,
        metadata={"response_id": data.get("id", "")},
    )


def _attach_parsed(response: LLMResponse, spec: StructuredOutputSpec) -> LLMResponse:
    try:
        parsed = json.loads(response.content) if response.content.strip() else None
    except json.JSONDecodeError as exc:
        raise InvalidStructuredOutputError(
            f"{response.provider}: structured output was not valid JSON: {exc}"
        ) from exc
    if spec.response_model is not None and parsed is not None:
        try:
            validated = spec.response_model.model_validate(parsed)
            parsed = validated.model_dump()
        except Exception as exc:  # noqa: BLE001
            raise InvalidStructuredOutputError(
                f"{response.provider}: structured output failed schema validation: {exc}"
            ) from exc
    return response.model_copy(
        update={"metadata": {**response.metadata, "parsed": parsed, "schema_name": spec.name}}
    )
