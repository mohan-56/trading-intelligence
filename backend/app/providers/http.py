"""Shared HTTP client with timeout, retry and honest error mapping."""

from __future__ import annotations

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.errors import ProviderError, ProviderRateLimited, ProviderTimeout
from app.core.logging import get_logger

log = get_logger("http")

USER_AGENT = "crypto-intelligence/0.1 (personal research use)"
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"},
            follow_redirects=True,
            timeout=httpx.Timeout(20.0, connect=10.0),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
    return _client


async def close_client() -> None:
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None


@retry(
    stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)), reraise=True,
)
async def _request(provider: str, url: str, *, params=None, timeout=None, headers=None) -> httpx.Response:
    return await get_client().get(url, params=params, timeout=timeout, headers=headers)


async def fetch(
    provider: str, url: str, *, params=None, timeout: float | None = None, headers: dict | None = None
) -> httpx.Response:
    try:
        response = await _request(provider, url, params=params, timeout=timeout, headers=headers)
    except httpx.TimeoutException as exc:
        raise ProviderTimeout(provider, f"timeout after retries: {exc}") from exc
    except httpx.HTTPError as exc:
        raise ProviderError(provider, f"transport error: {exc}") from exc

    if response.status_code == 429:
        raise ProviderRateLimited(provider, "rate limited (HTTP 429)")
    if response.status_code == 451:
        raise ProviderError(provider, "HTTP 451 -- unavailable in this jurisdiction")
    if response.status_code >= 500:
        raise ProviderError(provider, f"upstream error HTTP {response.status_code}")
    if response.status_code >= 400:
        raise ProviderError(provider, f"HTTP {response.status_code}: {response.text[:200]}")
    return response
