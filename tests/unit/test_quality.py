"""Data quality engine. Funding rate is the interesting case here: it's
legitimately negative, unlike every price series."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from app.core.provenance import Latency, Provenance
from app.domain.marketdata.models import Bar, BarSeries
from app.domain.marketdata.quality import Severity, check_series
from app.domain.symbols.models import AssetClass, Symbol, Timeframe

PROV = Provenance.single("test", Latency.EOD)
SYM = Symbol(ticker="BTCUSD", asset_class=AssetClass.CRYPTO, exchange="BINANCE")


def _series(closes: list[float]) -> BarSeries:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    bars = [Bar(ts=base + timedelta(days=i), open=c, high=c, low=c, close=c) for i, c in enumerate(closes)]
    return BarSeries.from_bars(SYM, Timeframe.D1, bars, PROV)


def _finding(report, check):
    return next((f for f in report.findings if f.check == check), None)


class TestNegativeValues:
    def test_negative_price_is_an_error(self):
        report = check_series(_series([100, 101, -5, 102]))
        finding = _finding(report, "non_positive_price")
        assert finding and finding.severity is Severity.ERROR
        assert not report.passed

    def test_negative_funding_rate_is_allowed(self):
        """Shorts pay longs when funding is negative -- a real, common state,
        not corrupt data."""
        report = check_series(_series([0.0001, -0.0003, 0.0002, -0.0001]), allow_negative=True)
        assert _finding(report, "non_positive_price") is None
        assert report.passed


class TestStructuralChecks:
    def test_empty_series_fails(self):
        report = check_series(BarSeries.empty(SYM, Timeframe.D1, PROV))
        assert not report.passed

    def test_duplicate_timestamps_fail(self):
        ts = datetime(2026, 1, 1, tzinfo=UTC)
        bars = [Bar(ts=ts, open=1, high=1, low=1, close=1)] * 2
        report = check_series(BarSeries.from_bars(SYM, Timeframe.D1, bars, PROV))
        assert not report.passed
        assert _finding(report, "duplicate_timestamps")

    def test_ohlc_violation_fails(self):
        bars = [Bar(ts=datetime(2026, 1, 1, tzinfo=UTC), open=10, high=5, low=8, close=9)]
        report = check_series(BarSeries.from_bars(SYM, Timeframe.D1, bars, PROV))
        assert not report.passed

    def test_clean_series_passes(self):
        report = check_series(_series([100 + i * 0.1 for i in range(120)]))
        assert report.passed and not report.errors


class TestAnomalyDetection:
    def test_real_crash_warns_but_never_quarantines(self):
        """MAD-based, not std-based: a crash day must not inflate its own
        threshold into invisibility. (Found on a prior build: an actual -40%
        day scored 11 sigma under std, under the 12-sigma threshold.)"""
        rng = np.random.default_rng(7)
        closes = list(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 120))))
        closes[80] = closes[79] * 0.55  # a -45% day, like BTC on 2020-03-12
        report = check_series(_series(closes))
        assert report.passed, "a real crash must never be quarantined"
        assert _finding(report, "extreme_move")

    def test_quiet_series_produces_no_flag(self):
        rng = np.random.default_rng(3)
        closes = list(100 * np.exp(np.cumsum(rng.normal(0, 0.015, 200))))
        assert _finding(check_series(_series(closes)), "extreme_move") is None
