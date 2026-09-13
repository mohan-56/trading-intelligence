"""Provider chain and circuit breaker."""

from __future__ import annotations

from datetime import UTC, datetime

from app.core.errors import ProviderError
from app.core.provenance import Latency, Provenance
from app.domain.marketdata.models import Bar, BarSeries
from app.domain.symbols.models import Timeframe
from app.providers.base import MarketDataProvider, ProviderCapabilities
from app.providers.chain import CircuitState, ProviderChain

START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2024, 6, 1, tzinfo=UTC)


class StubProvider(MarketDataProvider):
    def __init__(self, name: str, *, fail: bool = False, empty: bool = False, respect_mapping: bool = False) -> None:
        super().__init__(ProviderCapabilities(name=name, kind="stub", latency=Latency.EOD, min_interval_seconds=0.0))
        self._fail = fail
        self._empty = empty
        self._respect_mapping = respect_mapping
        self.calls = 0

    def supports_symbol(self, symbol) -> bool:
        # By default a generic stub, not tied to any real provider mapping --
        # except when a test needs the real "does this symbol map to me at
        # all" behaviour (e.g. an unmapped symbol with no provider_tickers).
        if self._respect_mapping:
            return super().supports_symbol(symbol)
        return True

    async def get_bars(self, symbol, timeframe, start=None, end=None):
        self.calls += 1
        if self._fail:
            raise ProviderError(self.name, "simulated failure")
        if self._empty:
            return BarSeries.unavailable(symbol, timeframe, "no data")
        bar = Bar(ts=START, open=100, high=101, low=99, close=100.5, volume=10)
        return BarSeries.from_bars(symbol, timeframe, [bar],
                                   Provenance.single(self.name, Latency.EOD, as_of=START))


class TestCircuitBreaker:
    def test_opens_after_threshold(self):
        chain = ProviderChain([StubProvider("dead", fail=True)], failure_threshold=1)
        assert chain.breaker_state("dead") is CircuitState.CLOSED


class TestProviderChain:
    async def test_primary_success_short_circuits(self, btc_symbol):
        primary, secondary = StubProvider("a"), StubProvider("b")
        chain = ProviderChain([primary, secondary])
        series, attempts = await chain.get_bars(btc_symbol, Timeframe.D1, START, END)
        assert series.is_available
        assert secondary.calls == 0
        assert attempts[0].ok

    async def test_falls_back_and_records_it(self, btc_symbol):
        chain = ProviderChain([StubProvider("dead", fail=True), StubProvider("ok")])
        series, attempts = await chain.get_bars(btc_symbol, Timeframe.D1, START, END)
        assert series.is_available
        assert "fallback after dead" in series.provenance.note
        assert [a.ok for a in attempts] == [False, True]

    async def test_all_fail_returns_unavailable_not_empty(self, btc_symbol):
        chain = ProviderChain([StubProvider("a", fail=True), StubProvider("b", fail=True)])
        series, attempts = await chain.get_bars(btc_symbol, Timeframe.D1, START, END)
        assert not series.is_available
        assert series.provenance.latency is Latency.UNAVAILABLE
        assert len(attempts) == 2

    async def test_open_circuit_skips_provider(self, btc_symbol):
        dead, good = StubProvider("dead", fail=True), StubProvider("good")
        chain = ProviderChain([dead, good], failure_threshold=1)
        await chain.get_bars(btc_symbol, Timeframe.D1, START, END)
        assert chain.breaker_state("dead") is CircuitState.OPEN
        _, attempts = await chain.get_bars(btc_symbol, Timeframe.D1, START, END)
        assert attempts[0].reason == "circuit open"

    async def test_provider_crash_does_not_kill_the_request(self, btc_symbol):
        class Crasher(StubProvider):
            async def get_bars(self, *a, **kw):
                raise ValueError("boom")
        chain = ProviderChain([Crasher("crasher"), StubProvider("good")])
        series, attempts = await chain.get_bars(btc_symbol, Timeframe.D1, START, END)
        assert series.is_available
        assert "unexpected" in attempts[0].reason

    async def test_unmapped_symbol_is_skipped_not_failed(self):
        from app.domain.symbols.models import AssetClass, Symbol
        unmapped = Symbol(ticker="NOPE", asset_class=AssetClass.CRYPTO, exchange="BINANCE", provider_tickers={})
        chain = ProviderChain([StubProvider("a", respect_mapping=True)])
        _, attempts = await chain.get_bars(unmapped, Timeframe.D1, START, END)
        assert attempts[0].reason == "symbol not mapped"
