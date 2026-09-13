"""Provenance is the honesty guarantee. If these pass, a simulated/stale
number cannot be laundered into a real-looking one by arithmetic."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.core.provenance import Latency, Provenance, Tracked

T0 = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)


class TestLatencyOrdering:
    def test_worst_wins_order(self):
        assert Latency.LIVE < Latency.DELAYED < Latency.EOD < Latency.HISTORICAL < Latency.UNAVAILABLE

    def test_unavailable_is_not_usable(self):
        assert Latency.EOD.is_usable
        assert not Latency.UNAVAILABLE.is_usable


class TestCombine:
    def test_live_plus_delayed_is_delayed(self):
        combined = Provenance.combine([
            Provenance.single("binance", Latency.LIVE, as_of=T0),
            Provenance.single("yahoo", Latency.DELAYED, as_of=T0 - timedelta(days=1)),
        ])
        assert combined.latency is Latency.DELAYED

    def test_unavailable_dominates(self):
        combined = Provenance.combine([
            Provenance.single("a", Latency.LIVE, as_of=T0),
            Provenance.unavailable("funding feed down"),
        ])
        assert combined.latency is Latency.UNAVAILABLE

    def test_as_of_takes_the_oldest(self):
        combined = Provenance.combine([
            Provenance.single("a", Latency.EOD, as_of=T0),
            Provenance.single("b", Latency.EOD, as_of=T0 - timedelta(days=5)),
        ])
        assert combined.as_of == T0 - timedelta(days=5)

    def test_empty_combination_is_unavailable(self):
        assert Provenance.combine([]).latency is Latency.UNAVAILABLE


class TestDegraded:
    def test_records_reason_and_marks_cached(self):
        p = Provenance.single("binance", Latency.EOD, as_of=T0)
        d = p.degraded("STALE CACHE: Binance rate limited", latency=Latency.HISTORICAL)
        assert d.is_cached
        assert d.latency is Latency.HISTORICAL
        assert "STALE CACHE" in d.note

    def test_degraded_never_improves_latency(self):
        p = Provenance.single("x", Latency.HISTORICAL)
        assert p.degraded("note", latency=Latency.LIVE).latency is Latency.HISTORICAL


class TestTracked:
    def test_derive_inherits_worst_provenance(self):
        a = Tracked(1.0, Provenance.single("a", Latency.LIVE, as_of=T0))
        b = Tracked(2.0, Provenance.single("b", Latency.DELAYED, as_of=T0))
        result = Tracked.derive(3.0, a, b)
        assert result.value == 3.0
        assert result.provenance.latency is Latency.DELAYED

    def test_derive_from_unavailable_input_is_unavailable(self):
        good = Tracked(1.0, Provenance.single("a", Latency.EOD, as_of=T0))
        missing = Tracked.missing("open interest feed down")
        result = Tracked.derive(0.5, good, missing)
        assert result.value is None
        assert not result.is_available
        assert "open interest feed down" in result.provenance.note

    def test_missing_is_none_not_zero(self):
        m: Tracked[float] = Tracked.missing("no data")
        assert m.value is None and m.value != 0.0
