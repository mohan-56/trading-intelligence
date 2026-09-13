"""Yahoo Finance chart API -- keyless context signals only (DXY, VIX, US10Y).

Not the primary data source for anything in this product; these three feed the
regime engine as macro/risk-appetite context alongside Binance's crypto data.
Talks to the chart endpoint directly (not the `yfinance` package) -- one less
dependency, we control parsing and error handling.

Verified working with a browser User-Agent; a plain programmatic UA gets
rejected on some edges.
"""

from __future__ import annotations

from datetime import UTC, datetime

from app.core.clock import utcnow
from app.core.logging import get_logger
from app.core.provenance import Latency, Provenance
from app.domain.marketdata.models import Bar, BarSeries
from app.domain.symbols.models import Symbol, Timeframe
from app.providers.base import MarketDataProvider
from app.providers.http import BROWSER_UA, fetch

log = get_logger("yahoo")

BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/"
_HEADERS = {"User-Agent": BROWSER_UA, "Accept": "application/json,text/plain,*/*"}


class YahooContextProvider(MarketDataProvider):
    async def get_bars(
        self, symbol: Symbol, timeframe: Timeframe,
        start: datetime | None = None, end: datetime | None = None,
    ) -> BarSeries:
        if timeframe is not Timeframe.D1:
            return BarSeries.unavailable(symbol, timeframe, "yahoo context is daily only")
        ticker = symbol.provider_ticker("yahoo")
        if not ticker:
            return BarSeries.unavailable(symbol, timeframe, "symbol not mapped for yahoo")

        await self._limiter.acquire()
        params: dict = {"interval": "1d"}
        if start or end:
            params["period1"] = int((start or datetime(1990, 1, 1, tzinfo=UTC)).timestamp())
            params["period2"] = int((end or utcnow()).timestamp())
        else:
            params["range"] = "max"

        response = await fetch(
            self.name, BASE_URL + ticker, params=params,
            timeout=self.capabilities.timeout_seconds, headers=_HEADERS,
        )
        payload = response.json()
        chart = payload.get("chart") or {}
        if chart.get("error"):
            return BarSeries.unavailable(symbol, timeframe, f"yahoo error for '{ticker}'")
        results = chart.get("result") or []
        if not results:
            return BarSeries.unavailable(symbol, timeframe, f"yahoo returned no result for '{ticker}'")

        bars = _parse_result(results[0])
        if not bars:
            return BarSeries.unavailable(symbol, timeframe, f"yahoo returned no usable bars for '{ticker}'")

        prov = Provenance.single(
            self.name, Latency.DELAYED, as_of=bars[-1].ts, fetched_at=utcnow(),
            note=f"yahoo chart API context signal ({ticker}); delayed, not for redistribution",
        )
        return BarSeries.from_bars(symbol, timeframe, bars, prov)


def _parse_result(result: dict) -> list[Bar]:
    stamps = result.get("timestamp") or []
    quotes = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    if not stamps or not quotes:
        return []
    opens, highs, lows, closes = quotes.get("open") or [], quotes.get("high") or [], quotes.get("low") or [], quotes.get("close") or []
    if not (len(opens) == len(highs) == len(lows) == len(closes) == len(stamps)):
        return []
    bars = []
    for i, epoch in enumerate(stamps):
        o, h, low_, c = opens[i], highs[i], lows[i], closes[i]
        if None in (o, h, low_, c):
            continue
        bar = Bar(ts=datetime.fromtimestamp(int(epoch), tz=UTC), open=float(o), high=float(h), low=float(low_), close=float(c))
        if bar.is_coherent():
            bars.append(bar)
    return bars
