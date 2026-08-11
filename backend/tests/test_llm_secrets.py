"""Tests for secret masking and observability in the LLM layer."""

from __future__ import annotations

import json

from loguru import logger

from app.llm.models import LLMMessage, LLMRequest, MessageRole
from app.llm.providers.ollama import OllamaClient
from tests.llm_helpers import OllamaMockTransport


def _capture_logs() -> tuple[list[str], object]:
    logs: list[str] = []

    sink_id = logger.add(
        lambda msg: logs.append(msg),
        level="DEBUG",
        format="{message} {extra}",
        diagnose=False,
    )
    return logs, sink_id


async def test_ollama_does_not_log_api_key() -> None:
    # Ollama has no API key, but ensure no Authorization-style data leaks.
    logs, sink_id = _capture_logs()
    try:
        mock = OllamaMockTransport()
        client = mock.make_client()
        await client.generate(
            LLMRequest(
                model="qwen2.5-coder:7b",
                messages=[LLMMessage(role=MessageRole.USER, content="secret-password-value")],
                metadata={"api_key": "sk-super-secret"},
            )
        )
        await client.close()
        blob = "\n".join(logs)
        # The prompt content and the api_key must not appear in logs.
        assert "secret-password-value" not in blob
        assert "sk-super-secret" not in blob
    finally:
        logger.remove(sink_id)


async def test_ollama_verbose_logging_includes_content_only_when_enabled() -> None:
    logs, sink_id = _capture_logs()
    try:
        mock = OllamaMockTransport()
        client = OllamaClient(
            base_url="http://localhost:11434",
            timeout=10.0,
            verbose_logging=False,
            http_client=__import__("httpx").AsyncClient(
                base_url="http://localhost:11434",
                transport=__import__("httpx").MockTransport(mock.handler),
                timeout=10.0,
            ),
        )
        await client.generate(
            LLMRequest(
                model="qwen2.5-coder:7b",
                messages=[LLMMessage(role=MessageRole.USER, content="ULTRA-SECRET-PROMPT")],
            )
        )
        await client.close()
        blob = "\n".join(logs)
        assert "ULTRA-SECRET-PROMPT" not in blob
    finally:
        logger.remove(sink_id)


async def test_openai_compat_authorization_not_logged() -> None:
    logs, sink_id = _capture_logs()
    try:
        from tests.llm_helpers import OpenAICompatMockTransport

        mock = OpenAICompatMockTransport()
        client = mock.make_client(api_key="sk-not-to-be-logged")
        await client.generate(
            LLMRequest(
                model="cloud-coder:32b",
                messages=[LLMMessage(role=MessageRole.USER, content="Hi")],
            )
        )
        await client.close()
        blob = "\n".join(logs)
        assert "sk-not-to-be-logged" not in blob
        assert "Bearer" not in blob
    finally:
        logger.remove(sink_id)


def test_llm_events_never_contain_prompts(captured_events) -> None:
    """Events carry provider/model/latency, never prompt contents."""
    import asyncio

    from app.events import EventType
    from tests.llm_helpers import OllamaMockTransport

    bus, events = captured_events
    mock = OllamaMockTransport()
    client = mock.make_client(event_bus=bus)

    async def run() -> None:
        await client.generate(
            LLMRequest(
                model="qwen2.5-coder:7b",
                messages=[LLMMessage(role=MessageRole.USER, content="DO-NOT-LEAK-THIS")],
            )
        )
        await client.close()

    asyncio.run(run())
    for e in events:
        blob = json.dumps(e.payload)
        assert "DO-NOT-LEAK-THIS" not in blob
        # Events should carry provider/model/request_id, not message content.
        if e.event_type in (EventType.LLM_REQUEST_STARTED, EventType.LLM_REQUEST_COMPLETED):
            assert "provider" in e.payload
            assert "model" in e.payload
