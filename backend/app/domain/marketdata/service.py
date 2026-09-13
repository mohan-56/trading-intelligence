"""Market data service. API requests read the lake only -- no network call
ever happens inside a request handler. Scheduled jobs call `refresh()`.
"""

from __future__ import annotations

from datetime import datetime

from app.core.logging import get_logger
from app.domain.marketdata.models import BarSeries
from app.domain.symbols.models import Symbol, Timeframe
from app.domain.symbols.registry import SymbolRegistry
from app.providers.chain import ChainAttempt, ProviderChain
from app.storage.lake import ParquetLake

log = get_logger("marketdata")


class MarketDataService:
    def __init__(self, registry: SymbolRegistry, chain: ProviderChain, lake: ParquetLake) -> None:
        self._registry = registry
        self._chain = chain
        self._lake = lake

    def get_bars(self, ticker: str, timeframe: Timeframe = Timeframe.D1,
                start: datetime | None = None, end: datetime | None = None) -> BarSeries:
        symbol = self._registry.get(ticker)
        return self._lake.read_bars(symbol, timeframe, start, end)

    def coverage(self) -> list[dict]:
        manifest = self._lake.manifest()
        out = []
        for symbol in self._registry.all():
            entry = manifest.get(symbol.ticker) or {}
            out.append({
                "ticker": symbol.ticker, "name": symbol.name, "asset_class": symbol.asset_class.value,
                "rows": entry.get("rows", 0), "start": entry.get("start"), "end": entry.get("end"),
                "provenance": entry.get("provenance") or {"latency": "UNAVAILABLE", "note": "not ingested yet", "sources": []},
            })
        return out

    async def refresh(
        self, symbol: Symbol, timeframe: Timeframe = Timeframe.D1,
        start: datetime | None = None, end: datetime | None = None, *, persist: bool = True,
    ) -> tuple[BarSeries, list[ChainAttempt]]:
        series, attempts = await self._chain.get_bars(symbol, timeframe, start, end)
        if persist and series.is_available:
            self._lake.write_bars(series)
        return series, attempts
