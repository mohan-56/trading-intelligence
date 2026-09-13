"""Risk engine tests. Sizing is deterministic and independent of any
strategy's opinion -- a strategy can be wrong about direction, but it can
never be the thing that decides how much is at stake."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from app.core.provenance import Latency, Provenance
from app.domain.backtest.models import MarketView, Signal
from app.domain.backtest.risk import RiskEngine
from app.domain.marketdata.models import Bar, BarSeries
from app.domain.symbols.models import AssetClass, Symbol, Timeframe

PROV = Provenance.single("test", Latency.EOD)
SYM = Symbol(ticker="BTCUSD", asset_class=AssetClass.CRYPTO, exchange="BINANCE")


def _view(closes: list[float]) -> MarketView:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    bars = [Bar(ts=base + timedelta(days=i), open=c, high=c, low=c, close=c) for i, c in enumerate(closes)]
    series = BarSeries.from_bars(SYM, Timeframe.D1, bars, PROV)
    return MarketView(now=base + timedelta(days=len(closes) - 1), series={"price": series})


class TestFlatIsAlwaysZero:
    def test_flat_signal_ignores_volatility_entirely(self):
        rng = np.random.default_rng(1)
        wild = list(100 * np.exp(np.cumsum(rng.normal(0, 0.1, 100))))
        engine = RiskEngine()
        assert engine.size(Signal.FLAT, _view(wild)) == 0.0


class TestVolTargetedSizing:
    def test_higher_realized_vol_gets_smaller_weight(self):
        rng = np.random.default_rng(2)
        calm = list(100 * np.exp(np.cumsum(rng.normal(0, 0.005, 60))))
        wild = list(100 * np.exp(np.cumsum(rng.normal(0, 0.05, 60))))
        engine = RiskEngine(target_vol_annual=0.40)
        calm_w = engine.size(Signal.LONG, _view(calm))
        wild_w = engine.size(Signal.LONG, _view(wild))
        assert calm_w > wild_w

    def test_weight_never_exceeds_max(self):
        rng = np.random.default_rng(3)
        very_calm = list(100 * np.exp(np.cumsum(rng.normal(0, 0.0001, 60))))
        engine = RiskEngine(target_vol_annual=0.40, max_weight=0.75)
        assert engine.size(Signal.LONG, _view(very_calm)) <= 0.75

    def test_weight_is_never_negative(self):
        rng = np.random.default_rng(4)
        prices = list(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 60))))
        engine = RiskEngine()
        assert engine.size(Signal.LONG, _view(prices)) >= 0.0


class TestDegenerateInputs:
    def test_not_enough_history_stays_flat(self):
        engine = RiskEngine(vol_lookback=21)
        assert engine.size(Signal.LONG, _view([100.0, 101.0, 102.0])) == 0.0

    def test_flat_price_series_zero_vol_stays_flat_not_infinite(self):
        """Zero volatility must never produce an infinite or huge weight --
        a division-by-zero here would size a position at 'unlimited'."""
        engine = RiskEngine()
        assert engine.size(Signal.LONG, _view([100.0] * 60)) == 0.0

    def test_unknown_signal_raises_rather_than_silently_sizing(self):
        engine = RiskEngine()
        with pytest.raises(ValueError):
            engine.size("BUY", _view([100.0] * 60))  # type: ignore[arg-type]
