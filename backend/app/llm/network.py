"""Network reachability checker for cloud provider fallback (Phase 5).

The :class:`ModelRouter` uses a :class:`NetworkChecker` as a *signal* (not a
rule) when considering cloud providers. Phase 2 shipped
:class:`AlwaysOfflineNetworkChecker` for deterministic offline tests.

Phase 5 adds :class:`HttpxNetworkChecker`, which performs a real HTTP probe
against a provider's base URL. It is used only for **cloud** providers —
local-first routing does not require a network check (local models are
preferred regardless of network status).

The checker never raises and never blocks startup. It has a short timeout so
a slow cloud endpoint does not stall routing.
"""

from __future__ import annotations

import httpx

from app.core.logging import get_logger
from app.llm.router import NetworkChecker

logger = get_logger("llm.network")

__all__ = ["HttpxNetworkChecker", "DEFAULT_PROBE_TIMEOUT"]

DEFAULT_PROBE_TIMEOUT = 5.0


class HttpxNetworkChecker(NetworkChecker):
    """Probes provider reachability with a short HTTP HEAD/GET request.

    A provider is considered reachable if the HTTP request completes with any
    status code (even 401/404 means the server is up). Connection errors and
    timeouts are treated as unreachable.
    """

    def __init__(
        self,
        *,
        probe_urls: dict[str, str] | None = None,
        timeout: float = DEFAULT_PROBE_TIMEOUT,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._probe_urls = probe_urls or {}
        self._timeout = timeout
        self._http = http_client
        self._owns_http = http_client is None

    async def is_available(self, provider_id: str) -> bool:
        url = self._probe_urls.get(provider_id)
        if not url:
            return False
        client = await self._client()
        try:
            resp = await client.get(url, timeout=self._timeout)
            # Any HTTP response means the server is reachable.
            return resp.status_code < 600
        except httpx.HTTPError:
            return False
        except Exception as exc:  # noqa: BLE001 - must never raise
            logger.bind(
                event="network.probe.error", provider=provider_id, error=type(exc).__name__
            ).debug("Network probe failed for '{}': {}", provider_id, str(exc))
            return False

    async def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self._timeout)
        return self._http

    async def close(self) -> None:
        if self._http is not None and self._owns_http:
            await self._http.aclose()
            self._http = None
