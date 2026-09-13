"""Strategy signal tests. Each strategy is a pure function of a MarketView --
tested directly, without running the full backtest engine."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from app.core.provenance import Latency, Provenance
from app.domain.backtest.models import MarketView, Signal
from app.domain.backtest.strategies import BuyAndHold, FundingContrarian, MeanReversion, Momentum
from app.domain.marketdata.models import Bar, BarSeries
from app.domain.symbols.models import AssetClass, Symbol, Timeframe

PROV = Provenance.single("test", Latency.EOD)
SYM = Symbol(ticker="BTCUSD", asset_class=AssetClass.CRYPTO, exchange="BINANCE")


def _price_series(closes: list[float]) -> BarSeries:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    bars = [Bar(ts=base + timedelta(days=i), open=c, high=c, low=c, close=c) for i, c in enumerate(closes)]
    return BarSeries.from_bars(SYM, Timeframe.D1, bars, PROV)


def _view(closes: list[float], funding: list[float] | None = None) -> MarketView:
    series = {"price": _price_series(closes)}
    if funding is not None:
        base = datetime(2024, 1, 1, tzinfo=UTC)
        fbars = [Bar(ts=base + timedelta(hours=8 * i), open=f, high=f, low=f, close=f) for i, f in enumerate(funding)]
        series["funding"] = BarSeries.from_bars(SYM, Timeframe.D1, fbars, PROV)
    return MarketView(now=datetime.now(UTC), series=series)


class TestBuyAndHold:
    def test_always_long(self):
        s = BuyAndHold()
        assert s.generate_signal(_view([1.0, 2.0])) is Signal.LONG
        assert s.generate_signal(_view([100.0])) is Signal.LONG


class TestMomentum:
    def test_long_when_trailing_return_is_positive(self):
        s = Momentum(lookback=10)
        rising = [100.0 + i for i in range(15)]
        assert s.generate_signal(_view(rising)) is Signal.LONG

    def test_flat_when_trailing_return_is_negative(self):
        s = Momentum(lookback=10)
        falling = [200.0 - i for i in range(15)]
        assert s.generate_signal(_view(falling)) is Signal.FLAT

    def test_flat_when_not_enough_history(self):
        s = Momentum(lookback=63)
        assert s.generate_signal(_view([100.0] * 10)) is Signal.FLAT


class TestMeanReversion:
    def test_long_after_a_sharp_drop(self):
        s = MeanReversion(lookback=5, z_threshold=-1.0, z_window=60)
        rng = np.random.default_rng(1)
        calm = list(100 * np.exp(np.cumsum(rng.normal(0, 0.005, 90))))
        crash = [*calm, calm[-1] * 0.7]  # a sharp one-day drop at the end
        assert s.generate_signal(_view(crash)) is Signal.LONG

    def test_flat_in_a_quiet_market(self):
        s = MeanReversion(lookback=5, z_threshold=-1.5, z_window=60)
        assert s.generate_signal(_view([100.0] * 90)) is Signal.FLAT

    def test_flat_when_not_enough_history(self):
        s = MeanReversion(lookback=10, z_window=90)
        assert s.generate_signal(_view([100.0] * 20)) is Signal.FLAT


class TestFundingContrarian:
    def test_flat_when_funding_series_absent(self):
        """Never silently falls back to a price-only heuristic -- if the
        signal it's named for isn't available, it does nothing."""
        s = FundingContrarian()
        assert s.generate_signal(_view([100.0] * 300)) is Signal.FLAT

    def test_long_when_funding_deeply_negative(self):
        s = FundingContrarian(z_threshold=-1.0, avg_window=5, z_window=60)
        rng = np.random.default_rng(2)
        calm_funding = list(rng.normal(0.0001, 0.00005, 90))
        crowded_short = calm_funding + [-0.01] * 6  # a sustained deeply negative stretch
        assert s.generate_signal(_view([100.0] * len(crowded_short), funding=crowded_short)) is Signal.LONG

    def test_flat_when_funding_history_too_short(self):
        s = FundingContrarian(avg_window=21, z_window=180)
        assert s.generate_signal(_view([100.0] * 50, funding=[0.0001] * 50)) is Signal.FLAT
