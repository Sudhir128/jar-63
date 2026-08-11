"""Ollama LLM provider (PRIMARY, local-first).

Talks to a local Ollama HTTP API (default ``http://localhost:11434``) using
``httpx``. No Ollama SDK is required. The base URL is configurable via
environment variables and never hardcoded in application logic.

Capabilities:
* chat generation (``/api/chat``)
* structured output (JSON ``format`` parameter)
* health check (``/api/tags`` + ``/api/show``)
* model discovery (``/api/tags``)
* model info (``/api/show``)
* timeout + structured error normalization

The provider never downloads, deletes, or modifies user models. If a model is
not installed, it reports ``MODEL_NOT_FOUND`` with guidance for the user.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from app.core.identifiers import generate_id, utc_now
from app.core.logging import get_logger
from app.events import Event, EventBus, EventType
from app.llm.client import HealthStatus, LLMClient, ProviderHealth
from app.llm.errors import (
    InvalidLLMResponseError,
    InvalidStructuredOutputError,
    LLMError,
    ModelNotFoundError,
)
from app.llm.models import (
    FinishReason,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMToolCall,
    LLMToolDefinition,
    LLMUsage,
    ModelCapability,
    ModelDefinition,
    StructuredOutputSpec,
)
from app.llm.providers.base import BaseProviderMixin, normalize_timeout

logger = get_logger("llm.ollama")

__all__ = ["OllamaClient", "OLLAMA_PROVIDER_ID"]

OLLAMA_PROVIDER_ID = "ollama"


class OllamaClient(BaseProviderMixin, LLMClient):
    """Ollama LLM client backed by the Ollama HTTP API."""

    provider_name = OLLAMA_PROVIDER_ID

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
        verbose_logging: bool = False,
        http_client: httpx.AsyncClient | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._default_timeout = timeout
        self._verbose = verbose_logging
        self._event_bus = event_bus
        # An injectable HTTP client enables offline testing without a real server.
        self._http = http_client
        self._owns_http = http_client is None

    async def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(base_url=self._base_url, timeout=self._default_timeout)
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
        try:
            client = await self._client()
            resp = await client.get("/api/tags", timeout=min(self._default_timeout, 10.0))
            if resp.status_code != 200:
                return ProviderHealth(
                    status=HealthStatus.INVALID_RESPONSE,
                    provider=self.provider_name,
                    detail=f"/api/tags returned status {resp.status_code}",
                )
            installed = {m.get("name", "") for m in resp.json().get("models", [])}
        except httpx.TimeoutException:
            return ProviderHealth(status=HealthStatus.TIMEOUT, provider=self.provider_name)
        except httpx.ConnectError:
            return ProviderHealth(status=HealthStatus.UNAVAILABLE, provider=self.provider_name)
        except Exception as exc:  # noqa: BLE001 - health checks must never raise
            return ProviderHealth(
                status=HealthStatus.INVALID_RESPONSE,
                provider=self.provider_name,
                detail=type(exc).__name__,
            )

        if model is None:
            return ProviderHealth(
                status=HealthStatus.AVAILABLE,
                provider=self.provider_name,
                detail=f"{len(installed)} model(s) installed",
            )
        if model in installed:
            return ProviderHealth(
                status=HealthStatus.AVAILABLE, provider=self.provider_name, model=model
            )
        return ProviderHealth(
            status=HealthStatus.MODEL_NOT_FOUND,
            provider=self.provider_name,
            model=model,
            detail=(
                f"Model '{model}' is not installed in Ollama. "
                f"Install it manually with: ollama pull {model}"
            ),
        )

    async def get_model_info(self, model: str) -> ModelDefinition | None:
        try:
            client = await self._client()
            resp = await client.post("/api/show", json={"name": model}, timeout=15.0)
        except httpx.HTTPError:
            return None
        if resp.status_code == 404:
            return None
        if resp.status_code != 200:
            return None
        data = resp.json()
        return _model_info_from_show(model, data)

    async def discover_models(self) -> list[str]:
        """Return the names of models installed in Ollama (best-effort)."""
        try:
            client = await self._client()
            resp = await client.get("/api/tags", timeout=15.0)
            if resp.status_code != 200:
                return []
            return [m.get("name", "") for m in resp.json().get("models", []) if m.get("name")]
        except httpx.HTTPError:
            return []

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
            payload["options"] = {"temperature": request.temperature}
        if request.max_tokens is not None:
            payload.setdefault("options", {})["num_predict"] = request.max_tokens
        if structured and spec is not None:
            payload["format"] = spec.to_json_schema()
        # Tool definitions: Ollama uses the same envelope as OpenAI
        # ({"type": "function", "function": {"name", "description", "parameters"}}).
        if request.tools:
            payload["tools"] = [_tool_def_to_ollama(t) for t in request.tools]

        started = time.monotonic()
        try:
            client = await self._client()
            resp = await client.post("/api/chat", json=payload, timeout=timeout)
        except Exception as exc:
            err = self._classify_transport_error(exc, provider=self.provider_name)
            await self._publish_failed(request, err)
            raise err from exc

        latency_ms = int((time.monotonic() - started) * 1000)

        if resp.status_code == 404:
            err = ModelNotFoundError(
                f"ollama: model '{request.model}' not found. "
                f"Install it manually with: ollama pull {request.model}"
            )
            await self._publish_failed(request, err)
            raise err
        if resp.status_code != 200:
            err = self._classify_http_status(resp.status_code, provider=self.provider_name)
            await self._publish_failed(request, err)
            raise err

        try:
            data = resp.json()
        except Exception as exc:
            err = InvalidLLMResponseError(f"ollama: invalid JSON response: {exc}")
            await self._publish_failed(request, err)
            raise err from exc

        response = _parse_chat_response(data, request.model, self.provider_name)
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
                    # The error message is safe (no secrets); include a short form.
                    "detail": str(error)[:300],
                },
                metadata=request.metadata,
            )
        )


def _msg_to_dict(m: LLMMessage) -> dict[str, Any]:
    """Convert an :class:`LLMMessage` to the Ollama chat message format.

    Handles three cases:
    * Plain text messages (system/user/assistant).
    * Assistant messages that carry tool calls (``tool_calls`` field).
    * Tool-role messages that respond to a tool call (``tool_call_id``).
    """
    d: dict[str, Any] = {"role": m.role.value, "content": m.content}
    if m.name:
        d["name"] = m.name
    if m.tool_calls:
        d["tool_calls"] = [_tool_call_to_ollama(tc) for tc in m.tool_calls]
    if m.tool_call_id:
        # Ollama expects tool results as role="tool" with the tool_call_id.
        d["tool_call_id"] = m.tool_call_id
    return d


def _tool_def_to_ollama(defn: LLMToolDefinition) -> dict[str, Any]:
    """Map an :class:`LLMToolDefinition` to Ollama's native tool format.

    Ollama uses the OpenAI-style envelope::

        {"type": "function", "function": {"name", "description", "parameters"}}
    """
    return {
        "type": "function",
        "function": {
            "name": defn.name,
            "description": defn.description,
            "parameters": defn.parameters or {"type": "object", "properties": {}},
        },
    }


def _tool_call_to_ollama(tc: LLMToolCall) -> dict[str, Any]:
    """Map an :class:`LLMToolCall` to Ollama's assistant-message tool_call format."""
    return {
        "id": tc.id,
        "type": "function",
        "function": {"name": tc.name, "arguments": dict(tc.arguments)},
    }


def _parse_tool_calls(raw: list[dict[str, Any]]) -> list[LLMToolCall]:
    """Parse Ollama's ``message.tool_calls`` into typed :class:`LLMToolCall`.

    Ollama returns ``arguments`` as a dict (not a JSON string). Malformed
    entries are skipped — they never reach the :class:`ToolExecutor`.
    """
    calls: list[LLMToolCall] = []
    for raw_tc in raw or []:
        func = raw_tc.get("function") or {}
        name = func.get("name")
        if not name:
            continue
        args = func.get("arguments")
        if args is None:
            args = {}
        elif isinstance(args, str):
            # Some models return arguments as a JSON string; parse it safely.
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                args = {}
        if not isinstance(args, dict):
            args = {}
        call_id = raw_tc.get("id") or generate_id("call")
        calls.append(LLMToolCall(id=call_id, name=name, arguments=args))
    return calls


def _parse_chat_response(data: dict[str, Any], model: str, provider: str) -> LLMResponse:
    msg = data.get("message") or {}
    content = msg.get("content", "") or ""
    raw_tool_calls = msg.get("tool_calls") or []
    tool_calls = _parse_tool_calls(raw_tool_calls)
    finish = data.get("done", True)
    if tool_calls:
        finish_reason = FinishReason.TOOL_CALLS
    elif finish:
        finish_reason = FinishReason.STOP
    else:
        finish_reason = FinishReason.LENGTH
    usage = LLMUsage(
        prompt_tokens=data.get("prompt_eval_count"),
        completion_tokens=data.get("eval_count"),
        total_tokens=((data.get("prompt_eval_count") or 0) + (data.get("eval_count") or 0))
        if (data.get("prompt_eval_count") is not None or data.get("eval_count") is not None)
        else None,
    )
    return LLMResponse(
        content=content,
        model=data.get("model", model),
        provider=provider,
        usage=usage,
        finish_reason=finish_reason,
        tool_calls=tool_calls,
        metadata={"created_at_iso": utc_now().isoformat()},
    )


def _attach_parsed(response: LLMResponse, spec: StructuredOutputSpec) -> LLMResponse:
    """Parse the response content as JSON and validate against the spec."""
    try:
        parsed = json.loads(response.content) if response.content.strip() else None
    except json.JSONDecodeError as exc:
        raise InvalidStructuredOutputError(
            f"ollama: structured output was not valid JSON: {exc}"
        ) from exc

    if spec.response_model is not None and parsed is not None:
        try:
            validated = spec.response_model.model_validate(parsed)
            parsed = validated.model_dump()
        except Exception as exc:  # noqa: BLE001 - validation failure is typed
            raise InvalidStructuredOutputError(
                f"ollama: structured output failed schema validation: {exc}"
            ) from exc

    return response.model_copy(
        update={"metadata": {**response.metadata, "parsed": parsed, "schema_name": spec.name}}
    )


def _model_info_from_show(model: str, data: dict[str, Any]) -> ModelDefinition:
    caps: set[ModelCapability] = {ModelCapability.CHAT}
    info = data.get("model_info") or data.get("details") or {}
    families = info.get("families") or []
    family_set = {str(f).lower() for f in families} if isinstance(families, list) else set()
    if any("bert" not in f for f in family_set) and family_set:
        caps.add(ModelCapability.CHAT)
    # Heuristic capability flags; refined in later phases.
    caps.add(ModelCapability.CODING)
    # Detect tool-calling support from capabilities reported by Ollama.
    capabilities = info.get("capabilities") or data.get("capabilities") or []
    cap_set = {str(c).lower() for c in capabilities} if isinstance(capabilities, list) else set()
    supports_tools = "tools" in cap_set
    if supports_tools:
        caps.add(ModelCapability.TOOL_CALLING)
    return ModelDefinition(
        model_id=model,
        provider=OLLAMA_PROVIDER_ID,
        display_name=model,
        capabilities=caps,
        context_window=info.get("context_length") or (8192 if "coder" in model else 4096),
        supports_structured_output=True,
        supports_tools=supports_tools,
        local=True,
        enabled=True,
    )
