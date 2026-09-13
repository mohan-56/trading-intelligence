"""State vector tests.

The alignment and causality tests are the important ones. A feature store
that lines up two instruments on the wrong calendar day, or leaks future
information, produces a regime engine that looks prescient in backtest and
is worthless live -- and nothing about the output looks wrong.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from app.core.provenance import Latency, Provenance
from app.domain.marketdata.models import Bar, BarSeries
from app.domain.quant.features import StateVectorBuilder, _forward_fill_onto, latest_snapshot
from app.domain.symbols.models import AssetClass, Symbol, Timeframe

PROV = Provenance.single("test", Latency.EOD)


def _dates(n, start=datetime(2020, 1, 1, tzinfo=UTC), step_days=1):
    return np.array([np.datetime64((start + timedelta(days=i * step_days)).replace(tzinfo=None), "s") for i in range(n)])


def _series(ticker, closes, step_days=1):
    base = datetime(2020, 1, 1, tzinfo=UTC)
    bars = [Bar(ts=base + timedelta(days=i * step_days), open=c, high=c, low=c, close=c) for i, c in enumerate(closes)]
    sym = Symbol(ticker=ticker, asset_class=AssetClass.CRYPTO, exchange="BINANCE")
    return BarSeries.from_bars(sym, Timeframe.D1, bars, PROV)


class FakeLake:
    def __init__(self, series_map: dict[str, BarSeries]) -> None:
        self._map = series_map

    def read_bars(self, symbol, timeframe=Timeframe.D1):
        if symbol.ticker in self._map:
            return self._map[symbol.ticker]
        sym = Symbol(ticker=symbol.ticker, asset_class=AssetClass.CRYPTO, exchange="BINANCE")
        return BarSeries.unavailable(sym, timeframe, f"{symbol.ticker} not in fake lake")


class FakeRegistry:
    """Resolves any ticker to a bare Symbol -- tests don't need it to exist
    in the real crypto universe.yaml, only in the fake lake's data."""

    def get(self, ticker: str) -> Symbol:
        return Symbol(ticker=ticker, asset_class=AssetClass.CRYPTO, exchange="TEST")


class TestAsOfJoin:
    def test_uses_last_value_at_or_before_target(self):
        target, source = _dates(5), _dates(3, step_days=2)
        out = _forward_fill_onto(target, source, np.array([10.0, 20.0, 30.0]), 30)
        assert list(out) == [10.0, 10.0, 20.0, 20.0, 30.0]

    def test_never_reaches_forward(self):
        target = _dates(5)
        source = _dates(1, start=datetime(2020, 1, 4, tzinfo=UTC))
        out = _forward_fill_onto(target, source, np.array([99.0]), 30)
        assert np.isnan(out[:3]).all()
        assert out[3] == 99.0

    def test_staleness_bound_stops_the_fill(self):
        target, source = _dates(60), _dates(1)
        out = _forward_fill_onto(target, source, np.array([5.0]), max_staleness_days=10)
        assert out[10] == 5.0
        assert np.isnan(out[11:]).all()


class TestRelativeStrengthAlignment:
    """Regression guard: SOLUSD has fewer rows than BTCUSD (listed later on
    Binance). Computing rel_strength without aligning them first either
    crashes on a length mismatch or -- worse -- silently pairs index i of one
    with index i of the other, comparing prices from different calendar days.
    This is exactly the bug the first real state-vector build hit."""

    def _cfg(self, features):
        return {"version": 1, "calendar_symbol": "BTC", "guards": {"min_history_days": 5}, "features": features}

    def test_shorter_series_does_not_crash(self):
        btc = _series("BTC", np.linspace(100, 200, 100))          # 100 rows
        sol = _series("SOL", np.linspace(10, 20, 40), step_days=1)  # only 40 rows, same start date
        lake = FakeLake({"BTC": btc, "SOL": sol})
        cfg = self._cfg([{"name": "sol_vs_btc", "source": "SOL", "transform": "rel_strength", "vs": "BTC", "window": 10}])

        matrix = StateVectorBuilder({"bars": lake}, cfg, registry=FakeRegistry()).build()
        assert matrix.raw.shape == (100, 1)
        # Past SOL's last real date, the feature must go NaN (stale, not fabricated).
        assert np.isnan(matrix.raw[99, 0])
        assert np.isfinite(matrix.raw[39, 0])

    def test_alignment_uses_correct_calendar_day_not_index(self):
        """If day 0 of BTC (100.0) were compared to day 0 of SOL's OWN short
        series instead of SOL's value on that calendar day, this would silently
        misalign. Construct SOL starting days LATER than BTC's index 0 and
        confirm the join respects real dates, not array position."""
        btc = _series("BTC", np.linspace(100, 300, 60))
        # SOL starts 20 calendar days after BTC's t=0, at a distinctive level.
        sol_start = datetime(2020, 1, 21, tzinfo=UTC)
        sol_bars = [Bar(ts=sol_start + timedelta(days=i), open=50 + i, high=50 + i, low=50 + i, close=50 + i) for i in range(30)]
        sol_sym = Symbol(ticker="SOL", asset_class=AssetClass.CRYPTO, exchange="BINANCE")
        sol = BarSeries.from_bars(sol_sym, Timeframe.D1, sol_bars, PROV)
        lake = FakeLake({"BTC": btc, "SOL": sol})
        cfg = self._cfg([{"name": "sol_vs_btc", "source": "SOL", "transform": "rel_strength", "vs": "BTC", "window": 5}])

        matrix = StateVectorBuilder({"bars": lake}, cfg, registry=FakeRegistry()).build()
        # Before SOL listed (BTC's first 20 rows), the feature must be NaN --
        # not silently computed against BTC's own early index-aligned values.
        assert np.isnan(matrix.raw[:20, 0]).all()
        assert np.isfinite(matrix.raw[25, 0])


class TestBuilder:
    def _cfg(self, features):
        return {"version": 99, "calendar_symbol": "SPX", "guards": {"min_history_days": 20}, "features": features}

    def test_builds_aligned_matrix(self):
        lake = FakeLake({"SPX": _series("SPX", np.linspace(100, 200, 300)), "VIX": _series("VIX", np.linspace(20, 15, 300))})
        cfg = self._cfg([
            {"name": "spx_mom", "source": "SPX", "transform": "log_return", "window": 20},
            {"name": "vix_level", "source": "VIX", "transform": "level"},
        ])
        m = StateVectorBuilder({"bars": lake}, cfg, registry=FakeRegistry()).build()
        assert len(m) == 300
        assert m.names == ("spx_mom", "vix_level")
        assert m.version == 99

    def test_missing_source_degrades_without_crashing(self):
        lake = FakeLake({"SPX": _series("SPX", np.linspace(100, 200, 100))})
        cfg = self._cfg([
            {"name": "spx_level", "source": "SPX", "transform": "level"},
            {"name": "ghost", "source": "NOPE", "transform": "level"},
        ])
        m = StateVectorBuilder({"bars": lake}, cfg, registry=FakeRegistry()).build()
        assert np.isfinite(m.raw[:, 0]).any()
        assert np.isnan(m.raw[:, 1]).all()
        assert not m.complete.any()
        assert "NOPE" in m.provenance.note

    def test_series_kind_routes_to_the_right_lake(self):
        bars_lake = FakeLake({"SPX": _series("SPX", np.linspace(100, 200, 100))})
        funding_lake = FakeLake({"SPX": _series("SPX", np.linspace(0.0001, 0.0002, 100))})
        cfg = self._cfg([
            {"name": "px", "source": "SPX", "series_kind": "bars", "transform": "level"},
            {"name": "fund", "source": "SPX", "series_kind": "funding", "transform": "level"},
        ])
        m = StateVectorBuilder({"bars": bars_lake, "funding": funding_lake}, cfg, registry=FakeRegistry()).build()
        assert m.raw[-1, 0] == pytest.approx(200.0, rel=0.05)
        assert m.raw[-1, 1] == pytest.approx(0.0002, rel=0.05)


class TestCausality:
    def test_truncating_history_does_not_change_earlier_rows(self):
        closes = 100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.02, 300)))
        cfg = {"version": 1, "calendar_symbol": "SPX", "guards": {"min_history_days": 30},
               "features": [{"name": "mom", "source": "SPX", "transform": "log_return", "window": 20}]}
        full = StateVectorBuilder({"bars": FakeLake({"SPX": _series("SPX", closes)})}, cfg, registry=FakeRegistry()).build()
        cut = StateVectorBuilder({"bars": FakeLake({"SPX": _series("SPX", closes[:200])})}, cfg, registry=FakeRegistry()).build()
        a, b = full.z[:200], cut.z
        both = np.isfinite(a) & np.isfinite(b)
        assert both.sum() > 100
        np.testing.assert_allclose(a[both], b[both], rtol=1e-8, atol=1e-8)


class TestSnapshot:
    def test_snapshot_is_as_of_the_requested_date(self):
        lake = FakeLake({"SPX": _series("SPX", np.linspace(100, 200, 200))})
        cfg = {"version": 1, "calendar_symbol": "SPX", "guards": {"min_history_days": 20},
               "features": [{"name": "lvl", "source": "SPX", "transform": "level"}]}
        m = StateVectorBuilder({"bars": lake}, cfg, registry=FakeRegistry()).build()
        snap = latest_snapshot(m, datetime(2020, 3, 1, tzinfo=UTC))
        assert snap["available"]
        assert snap["date"] <= "2020-03-01"


class TestRealConfig:
    def test_shipped_config_parses(self):
        from app.core.config import features_config
        from app.domain.quant.features import FeatureSpec

        specs = [FeatureSpec.from_dict(r) for r in features_config()["features"]]
        assert len(specs) >= 10
        assert len({s.name for s in specs}) == len(specs)

    def test_shipped_config_covers_price_funding_and_macro(self):
        from app.core.config import features_config

        kinds = {r.get("series_kind", "bars") for r in features_config()["features"]}
        assert {"bars", "funding"} <= kinds

    def test_guards_forbid_full_sample_standardization(self):
        from app.core.config import features_config

        assert features_config()["guards"]["standardization"] == "expanding"
