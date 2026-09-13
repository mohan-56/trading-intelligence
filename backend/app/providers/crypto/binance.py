"""Binance -- primary data source. No API key for public market data.

Two APIs:
  api.binance.com   spot klines (OHLCV) -- what the regime engine's price data is
  fapi.binance.com  USD-M futures: funding rate, open interest -- the crypto-
                     specific regime signals nothing else in this product has

Funding rate settles every 8 hours (00:00, 08:00, 16:00 UTC). We store it as a
level series (open=high=low=close=rate) timestamped at settlement -- never
averaged in before it actually printed (docs principle: no look-ahead).
"""

from __future__ import annotations

from datetime import UTC, datetime

from app.core.clock import utcnow
from app.core.logging import get_logger
from app.core.provenance import Latency, Provenance
from app.domain.marketdata.models import Bar, BarSeries
from app.domain.symbols.models import Symbol, Timeframe
from app.providers.base import MarketDataProvider
from app.providers.http import fetch

log = get_logger("binance")

SPOT_URL = "https://api.binance.com/api/v3/klines"
FUTURES_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
FUTURES_OI_HIST_URL = "https://fapi.binance.com/futures/data/openInterestHist"
MAX_LIMIT = 1000

_INTERVALS = {
    Timeframe.D1: "1d", Timeframe.H1: "1h", Timeframe.M15: "15m",
    Timeframe.M5: "5m", Timeframe.M1: "1m",
}


class BinanceProvider(MarketDataProvider):
    """Spot OHLCV. Paginated -- Binance caps each response at 1000 klines."""

    async def get_bars(
        self, symbol: Symbol, timeframe: Timeframe,
        start: datetime | None = None, end: datetime | None = None,
    ) -> BarSeries:
        interval = _INTERVALS.get(timeframe)
        if interval is None:
            return BarSeries.unavailable(symbol, timeframe, f"binance has no interval for {timeframe}")
        ticker = symbol.provider_ticker("binance")
        if not ticker:
            return BarSeries.unavailable(symbol, timeframe, "symbol not mapped for binance")

        bars: list[Bar] = []
        cursor_ms = int(start.timestamp() * 1000) if start else None
        end_ms = int(end.timestamp() * 1000) if end else None

        while True:
            await self._limiter.acquire()
            params: dict = {"symbol": ticker, "interval": interval, "limit": MAX_LIMIT}
            if cursor_ms is not None:
                params["startTime"] = cursor_ms
            if end_ms is not None:
                params["endTime"] = end_ms

            payload = (
                await fetch(self.name, SPOT_URL, params=params, timeout=self.capabilities.timeout_seconds)
            ).json()
            if not payload:
                break
            page = [b for b in (_to_bar(k) for k in payload) if b is not None and b.is_coherent()]
            bars.extend(page)

            if len(payload) < MAX_LIMIT:
                break
            next_open = int(payload[-1][0]) + 1
            if cursor_ms is not None and next_open <= cursor_ms:
                break
            cursor_ms = next_open
            if end_ms is not None and cursor_ms >= end_ms:
                break

        if not bars:
            return BarSeries.unavailable(symbol, timeframe, f"binance returned no klines for '{ticker}'")

        latency = Latency.LIVE if timeframe is not Timeframe.D1 else Latency.EOD
        prov = Provenance.single(
            self.name, latency, as_of=bars[-1].ts, fetched_at=utcnow(),
            note=f"binance spot klines ({ticker} {interval})",
        )
        return BarSeries.from_bars(symbol, timeframe, bars, prov)


class BinanceFundingProvider(MarketDataProvider):
    """Perpetual funding rate -- the clearest positioning signal free data
    gives you. Positive = longs pay shorts (crowded long); negative = the
    reverse. Settles every 8 hours; timestamped at settlement, not query time.
    """

    def supports_symbol(self, symbol: Symbol) -> bool:
        # This and BinanceOpenInterestProvider read a shared "binance_futures"
        # mapping, not one keyed by self.name -- the base class default check
        # (symbol.provider_ticker(self.name)) rejected every symbol before
        # get_bars ever ran. Found by the first real backfill: 5/5 "failed"
        # with a reason that turned out to be this, not a real API problem.
        return symbol.provider_ticker("binance_futures") is not None

    async def get_bars(
        self, symbol: Symbol, timeframe: Timeframe,
        start: datetime | None = None, end: datetime | None = None,
    ) -> BarSeries:
        ticker = symbol.provider_ticker("binance_futures")
        if not ticker:
            return BarSeries.unavailable(symbol, timeframe, "symbol not mapped for binance futures")

        bars: list[Bar] = []
        cursor_ms = int(start.timestamp() * 1000) if start else None
        end_ms = int(end.timestamp() * 1000) if end else None

        while True:
            await self._limiter.acquire()
            params: dict = {"symbol": ticker, "limit": MAX_LIMIT}
            if cursor_ms is not None:
                params["startTime"] = cursor_ms
            if end_ms is not None:
                params["endTime"] = end_ms

            payload = (
                await fetch(self.name, FUTURES_FUNDING_URL, params=params, timeout=self.capabilities.timeout_seconds)
            ).json()
            if not payload:
                break
            for row in payload:
                try:
                    ts = datetime.fromtimestamp(int(row["fundingTime"]) / 1000, tz=UTC)
                    rate = float(row["fundingRate"])
                except (KeyError, ValueError, TypeError):
                    continue
                # Funding rate is legitimately negative (shorts pay longs) --
                # never flagged as a data quality error, unlike a negative price.
                bars.append(Bar(ts=ts, open=rate, high=rate, low=rate, close=rate, volume=0.0))

            if len(payload) < MAX_LIMIT:
                break
            next_open = int(payload[-1]["fundingTime"]) + 1
            if cursor_ms is not None and next_open <= cursor_ms:
                break
            cursor_ms = next_open
            if end_ms is not None and cursor_ms >= end_ms:
                break

        if not bars:
            return BarSeries.unavailable(symbol, timeframe, f"binance returned no funding history for '{ticker}'")
        prov = Provenance.single(
            self.name, Latency.EOD, as_of=bars[-1].ts, fetched_at=utcnow(),
            note=f"binance perpetual funding rate ({ticker}); timestamped at 8h settlement",
        )
        return BarSeries.from_bars(symbol, timeframe, bars, prov)


class BinanceOpenInterestProvider(MarketDataProvider):
    """Open interest history. Binance's public history endpoint only retains
    ~30 days -- fine for the live regime signal, not for a 15-year backtest
    input. We treat it as context, not a backtested feature input for now.
    """

    def supports_symbol(self, symbol: Symbol) -> bool:
        return symbol.provider_ticker("binance_futures") is not None

    async def get_bars(
        self, symbol: Symbol, timeframe: Timeframe,
        start: datetime | None = None, end: datetime | None = None,
    ) -> BarSeries:
        ticker = symbol.provider_ticker("binance_futures")
        if not ticker:
            return BarSeries.unavailable(symbol, timeframe, "symbol not mapped for binance futures")

        await self._limiter.acquire()
        params = {"symbol": ticker, "period": "1d", "limit": 30}
        payload = (
            await fetch(self.name, FUTURES_OI_HIST_URL, params=params, timeout=self.capabilities.timeout_seconds)
        ).json()

        bars: list[Bar] = []
        for row in payload or []:
            try:
                ts = datetime.fromtimestamp(int(row["timestamp"]) / 1000, tz=UTC)
                oi = float(row["sumOpenInterestValue"])
            except (KeyError, ValueError, TypeError):
                continue
            bars.append(Bar(ts=ts, open=oi, high=oi, low=oi, close=oi, volume=0.0))

        if not bars:
            return BarSeries.unavailable(symbol, timeframe, f"binance returned no OI history for '{ticker}'")
        prov = Provenance.single(
            self.name, Latency.EOD, as_of=bars[-1].ts, fetched_at=utcnow(),
            note=f"binance open interest, USD notional ({ticker}); ~30d retention",
        )
        return BarSeries.from_bars(symbol, timeframe, bars, prov)


def _to_bar(kline: list) -> Bar | None:
    try:
        return Bar(
            ts=datetime.fromtimestamp(int(kline[0]) / 1000, tz=UTC),
            open=float(kline[1]), high=float(kline[2]), low=float(kline[3]),
            close=float(kline[4]), volume=float(kline[5]),
        )
    except (ValueError, IndexError, TypeError):
        return None
