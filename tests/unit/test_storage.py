"""Parquet lake round-trip, idempotency, cache invalidation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.core.provenance import Latency, Provenance
from app.domain.marketdata.models import Bar, BarSeries
from app.domain.symbols.models import Timeframe

PROV = Provenance.single("binance", Latency.EOD)


def _series(symbol, n=200, seed=1.0):
    import numpy as np
    rng = np.random.default_rng(int(seed))
    base = datetime(2024, 1, 1, tzinfo=UTC)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    bars = [
        Bar(ts=base + timedelta(days=i), open=float(c), high=float(c) + 1, low=float(c) - 1, close=float(c), volume=10.0)
        for i, c in enumerate(closes)
    ]
    return BarSeries.from_bars(symbol, Timeframe.D1, bars, PROV)


class TestRoundTrip:
    def test_write_then_read_preserves_prices(self, tmp_lake, btc_symbol):
        series = _series(btc_symbol)
        tmp_lake.write_bars(series)
        loaded = tmp_lake.read_bars(btc_symbol, Timeframe.D1)
        assert len(loaded) == len(series)
        import pytest
        assert loaded.close == pytest.approx(series.close)

    def test_provenance_survives(self, tmp_lake, btc_symbol):
        tmp_lake.write_bars(_series(btc_symbol))
        loaded = tmp_lake.read_bars(btc_symbol, Timeframe.D1)
        assert loaded.provenance.latency is Latency.EOD
        assert "binance" in loaded.provenance.sources
        assert loaded.provenance.is_cached

    def test_single_file_per_symbol(self, tmp_lake, btc_symbol):
        tmp_lake.write_bars(_series(btc_symbol))
        files = list(tmp_lake.series_dir(btc_symbol, Timeframe.D1).glob("*.parquet"))
        assert [f.name for f in files] == ["data.parquet"]


class TestIdempotency:
    def test_rewriting_same_data_does_not_duplicate(self, tmp_lake, btc_symbol):
        series = _series(btc_symbol)
        tmp_lake.write_bars(series)
        tmp_lake.write_bars(series)
        assert len(tmp_lake.read_bars(btc_symbol, Timeframe.D1)) == len(series)

    def test_write_invalidates_cache(self, tmp_lake, btc_symbol):
        early = _series(btc_symbol, n=100, seed=9)
        tmp_lake.write_bars(early)
        assert len(tmp_lake.read_bars(btc_symbol, Timeframe.D1)) == 100

        later = _series(btc_symbol, n=150, seed=9)
        tmp_lake.write_bars(later)
        reread = tmp_lake.read_bars(btc_symbol, Timeframe.D1)
        assert len(reread) > 100, "cache served stale data after a write"


class TestManifest:
    def test_records_what_we_hold(self, tmp_lake, btc_symbol):
        assert tmp_lake.manifest() == {}
        series = _series(btc_symbol)
        tmp_lake.write_bars(series)
        entry = tmp_lake.manifest()[btc_symbol.ticker]
        assert entry["rows"] == len(series)
        assert entry["provenance"]["latency"] == "EOD"


class TestMissingData:
    def test_unknown_symbol_reads_as_unavailable(self, tmp_lake, btc_symbol):
        loaded = tmp_lake.read_bars(btc_symbol, Timeframe.D1)
        assert not loaded.is_available
        assert loaded.provenance.latency is Latency.UNAVAILABLE
