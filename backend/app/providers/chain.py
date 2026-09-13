"""Provider chain: primary -> fallback -> cache -> UNAVAILABLE.

A chain never silently downgrades. Falling back mutates the Provenance and the
UI surfaces it -- "showing cached data because Binance failed" is a feature;
showing stale data as current is the failure this exists to prevent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from app.core.errors import ProviderError
from app.core.logging import get_logger
from app.domain.marketdata.models import BarSeries
from app.domain.symbols.models import Symbol, Timeframe
from app.providers.base import MarketDataProvider

log = get_logger("chain")


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    failure_threshold: int = 3
    open_seconds: float = 300.0
    _failures: int = field(default=0, init=False)
    _opened_at: float = field(default=0.0, init=False)
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)

    @property
    def state(self) -> CircuitState:
        if self._state is CircuitState.OPEN and (time.monotonic() - self._opened_at >= self.open_seconds):
            self._state = CircuitState.HALF_OPEN
        return self._state

    @property
    def allows_request(self) -> bool:
        return self.state is not CircuitState.OPEN

    def record_success(self) -> None:
        self._failures = 0
        self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()


@dataclass(frozen=True, slots=True)
class ChainAttempt:
    provider: str
    ok: bool
    reason: str = ""


class ProviderChain:
    def __init__(self, providers: list[MarketDataProvider], *, failure_threshold: int = 3, open_seconds: float = 300.0) -> None:
        self._providers = providers
        self._breakers = {p.name: CircuitBreaker(failure_threshold, open_seconds) for p in providers}

    @property
    def provider_names(self) -> list[str]:
        return [p.name for p in self._providers]

    def health(self) -> dict[str, str]:
        return {name: b.state.value for name, b in self._breakers.items()}

    def breaker_state(self, name: str) -> CircuitState:
        return self._breakers[name].state

    async def get_bars(
        self, symbol: Symbol, timeframe: Timeframe,
        start: datetime | None = None, end: datetime | None = None,
    ) -> tuple[BarSeries, list[ChainAttempt]]:
        attempts: list[ChainAttempt] = []
        for provider in self._providers:
            if not provider.supports_symbol(symbol):
                attempts.append(ChainAttempt(provider.name, False, "symbol not mapped"))
                continue
            breaker = self._breakers[provider.name]
            if not breaker.allows_request:
                attempts.append(ChainAttempt(provider.name, False, "circuit open"))
                continue

            try:
                series = await provider.get_bars(symbol, timeframe, start, end)
            except ProviderError as exc:
                breaker.record_failure()
                attempts.append(ChainAttempt(provider.name, False, str(exc)))
                log.warning("provider_failed", provider=provider.name, symbol=symbol.ticker, error=str(exc))
                continue
            except Exception as exc:
                breaker.record_failure()
                attempts.append(ChainAttempt(provider.name, False, f"unexpected: {exc}"))
                log.error("provider_crashed", provider=provider.name, symbol=symbol.ticker, error=str(exc))
                continue

            if series.is_available:
                breaker.record_success()
                attempts.append(ChainAttempt(provider.name, True))
                if attempts[0].provider != provider.name:
                    failed = ", ".join(a.provider for a in attempts if not a.ok)
                    series = _with_provenance(series, series.provenance.degraded(f"fallback after {failed} unavailable"))
                return series, attempts

            breaker.record_success()
            attempts.append(ChainAttempt(provider.name, False, "no data"))

        reason = "; ".join(f"{a.provider}: {a.reason}" for a in attempts) or "no providers configured"
        return BarSeries.unavailable(symbol, timeframe, f"all providers failed -- {reason}"), attempts


def _with_provenance(series: BarSeries, provenance) -> BarSeries:
    return BarSeries(
        symbol=series.symbol, timeframe=series.timeframe, ts=series.ts,
        open=series.open, high=series.high, low=series.low, close=series.close,
        volume=series.volume, provenance=provenance,
    )
