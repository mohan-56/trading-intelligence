"""Provider abstraction. Nothing in domain/ or api/ ever names a concrete
provider -- swapping Binance for a paid feed is a configs/providers.yaml edit.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

from app.core.provenance import Latency
from app.domain.marketdata.models import BarSeries
from app.domain.symbols.models import Symbol, Timeframe


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    name: str
    kind: str
    latency: Latency
    trust: int = 50
    timeframes: tuple[Timeframe, ...] = (Timeframe.D1,)
    min_interval_seconds: float = 1.0
    timeout_seconds: float = 20.0
    enabled: bool = True


class RateLimiter:
    """Minimum-interval throttle. Free providers are rate-limited by
    politeness as much as by policy."""

    def __init__(self, min_interval_seconds: float) -> None:
        self._min_interval = min_interval_seconds
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            elapsed = time.monotonic() - self._last
            wait = self._min_interval - elapsed
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()


class MarketDataProvider(ABC):
    def __init__(self, capabilities: ProviderCapabilities) -> None:
        self.capabilities = capabilities
        self._limiter = RateLimiter(capabilities.min_interval_seconds)

    @property
    def name(self) -> str:
        return self.capabilities.name

    def supports_symbol(self, symbol: Symbol) -> bool:
        return symbol.provider_ticker(self.name) is not None

    @abstractmethod
    async def get_bars(
        self, symbol: Symbol, timeframe: Timeframe,
        start: datetime | None = None, end: datetime | None = None,
    ) -> BarSeries: ...

    async def aclose(self) -> None:
        return None
