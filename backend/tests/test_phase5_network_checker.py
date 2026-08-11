"""Phase 5 tests: HttpxNetworkChecker for cloud fallback.

All tests are offline — they use httpx.MockTransport to simulate cloud
provider reachability. No real network access.
"""

from __future__ import annotations

import httpx

from app.llm.network import DEFAULT_PROBE_TIMEOUT, HttpxNetworkChecker


def _mock_transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


async def test_reachable_returns_true() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": []})

    transport = _mock_transport(handler)
    client = httpx.AsyncClient(transport=transport)
    checker = HttpxNetworkChecker(
        probe_urls={"openai_compatible": "https://cloud.example.com/models"},
        http_client=client,
    )
    assert await checker.is_available("openai_compatible") is True
    await client.aclose()


async def test_connection_error_returns_false() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    transport = _mock_transport(handler)
    client = httpx.AsyncClient(transport=transport)
    checker = HttpxNetworkChecker(
        probe_urls={"openai_compatible": "https://cloud.example.com/models"},
        http_client=client,
    )
    assert await checker.is_available("openai_compatible") is False
    await client.aclose()


async def test_timeout_returns_false() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    transport = _mock_transport(handler)
    client = httpx.AsyncClient(transport=transport)
    checker = HttpxNetworkChecker(
        probe_urls={"openai_compatible": "https://cloud.example.com/models"},
        http_client=client,
    )
    assert await checker.is_available("openai_compatible") is False
    await client.aclose()


async def test_unknown_provider_returns_false() -> None:
    checker = HttpxNetworkChecker(probe_urls={"openai_compatible": "https://x/models"})
    assert await checker.is_available("unknown_provider") is False


async def test_no_probe_url_returns_false() -> None:
    checker = HttpxNetworkChecker(probe_urls={})
    assert await checker.is_available("openai_compatible") is False


async def test_any_http_status_means_reachable() -> None:
    """Even 401/404 means the server is up."""
    for status in (401, 403, 404, 500):

        def handler(request: httpx.Request, status=status) -> httpx.Response:
            return httpx.Response(status)

        transport = _mock_transport(handler)
        client = httpx.AsyncClient(transport=transport)
        checker = HttpxNetworkChecker(
            probe_urls={"openai_compatible": "https://cloud.example.com/models"},
            http_client=client,
        )
        assert await checker.is_available("openai_compatible") is True, f"status {status}"
        await client.aclose()


async def test_never_raises_on_unexpected_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise ValueError("unexpected")

    transport = _mock_transport(handler)
    client = httpx.AsyncClient(transport=transport)
    checker = HttpxNetworkChecker(
        probe_urls={"openai_compatible": "https://cloud.example.com/models"},
        http_client=client,
    )
    # Must not raise.
    assert await checker.is_available("openai_compatible") is False
    await client.aclose()


def test_default_timeout_is_reasonable() -> None:
    assert DEFAULT_PROBE_TIMEOUT <= 10.0


async def test_close_releases_owned_client() -> None:
    checker = HttpxNetworkChecker(probe_urls={"p": "https://x"})
    await checker.is_available("p")  # forces client creation
    await checker.close()
