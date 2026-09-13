"""Regime engine tests. Deterministic given a FeatureMatrix -- every call
must trace back to the exact z-scores that produced it."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from app.core.provenance import Latency, Provenance
from app.domain.quant.features import FeatureMatrix
from app.domain.regime.engine import (
    AltRotation,
    Positioning,
    Trend,
    VolState,
    classify_history,
    classify_latest,
    classify_row,
)

NAMES = (
    "btc_trend_200d", "btc_mom_63d", "btc_vol_ratio", "btc_funding_z",
    "eth_vs_btc_63d", "sol_vs_btc_63d", "bnb_vs_btc_63d", "xrp_vs_btc_63d",
    "dxy_mom_63d", "vix_level", "us10y_chg_63d",
)


def _matrix(raw_row: dict, z_row: dict, n_rows: int = 1, version: int = 1) -> FeatureMatrix:
    dates = np.array([np.datetime64(f"2026-01-{i + 1:02d}") for i in range(n_rows)], dtype="datetime64[s]")
    raw = np.tile([raw_row.get(n, np.nan) for n in NAMES], (n_rows, 1))
    z = np.tile([z_row.get(n, np.nan) for n in NAMES], (n_rows, 1))
    complete = np.isfinite(raw).all(axis=1)
    prov = Provenance.single("test", Latency.EOD)
    return FeatureMatrix(dates=dates, names=NAMES, raw=raw, z=z, complete=complete, provenance=prov, version=version)


class TestTrendClassification:
    def test_clear_uptrend(self):
        raw = {"btc_trend_200d": 0.15, "btc_mom_63d": 0.20, "btc_vol_ratio": 1.0}
        z = {"btc_trend_200d": 1.2, "btc_mom_63d": 1.0}
        call = classify_row(_matrix(raw, z), 0)
        assert call.trend is Trend.UP

    def test_clear_downtrend(self):
        raw = {"btc_trend_200d": -0.15, "btc_mom_63d": -0.20, "btc_vol_ratio": 1.0}
        z = {"btc_trend_200d": -1.2, "btc_mom_63d": -1.0}
        call = classify_row(_matrix(raw, z), 0)
        assert call.trend is Trend.DOWN

    def test_near_zero_is_sideways_not_flip_flopping(self):
        """A z just barely past zero must not swing the whole label -- that's
        what the threshold exists to prevent."""
        raw = {"btc_trend_200d": 0.001, "btc_mom_63d": 0.001, "btc_vol_ratio": 1.0}
        z = {"btc_trend_200d": 0.05, "btc_mom_63d": 0.02}
        assert classify_row(_matrix(raw, z), 0).trend is Trend.SIDEWAYS

    def test_trend_and_momentum_disagreeing_is_sideways(self):
        """Price above its 200d average but currently falling -- not a clean
        uptrend, and the engine must not call it one."""
        raw = {"btc_trend_200d": 0.10, "btc_mom_63d": -0.05, "btc_vol_ratio": 1.0}
        z = {"btc_trend_200d": 1.0, "btc_mom_63d": -0.5}
        assert classify_row(_matrix(raw, z), 0).trend is Trend.SIDEWAYS


class TestVolatilityClassification:
    def test_expanding(self):
        raw = {"btc_vol_ratio": 1.5, "btc_trend_200d": 0, "btc_mom_63d": 0}
        assert classify_row(_matrix(raw, {}), 0).volatility is VolState.HIGH

    def test_compressed(self):
        raw = {"btc_vol_ratio": 0.6, "btc_trend_200d": 0, "btc_mom_63d": 0}
        assert classify_row(_matrix(raw, {}), 0).volatility is VolState.LOW

    def test_normal_band(self):
        raw = {"btc_vol_ratio": 1.0, "btc_trend_200d": 0, "btc_mom_63d": 0}
        assert classify_row(_matrix(raw, {}), 0).volatility is VolState.NORMAL


class TestPositioning:
    def test_crowded_long(self):
        raw = {"btc_funding_z": 0, "btc_trend_200d": 0, "btc_mom_63d": 0, "btc_vol_ratio": 1.0}
        z = {"btc_funding_z": 1.5}
        assert classify_row(_matrix(raw, z), 0).positioning is Positioning.CROWDED_LONG

    def test_crowded_short(self):
        raw = {"btc_funding_z": 0, "btc_trend_200d": 0, "btc_mom_63d": 0, "btc_vol_ratio": 1.0}
        z = {"btc_funding_z": -1.5}
        assert classify_row(_matrix(raw, z), 0).positioning is Positioning.CROWDED_SHORT


class TestAltRotation:
    def test_alt_season_when_alts_outrun_btc(self):
        raw = {"btc_trend_200d": 0, "btc_mom_63d": 0, "btc_vol_ratio": 1.0}
        z = {"eth_vs_btc_63d": 0.5, "sol_vs_btc_63d": 0.6, "bnb_vs_btc_63d": 0.4, "xrp_vs_btc_63d": 0.5}
        assert classify_row(_matrix(raw, z), 0).alt_rotation is AltRotation.ALT_SEASON

    def test_btc_dominance_when_alts_lag(self):
        raw = {"btc_trend_200d": 0, "btc_mom_63d": 0, "btc_vol_ratio": 1.0}
        z = {"eth_vs_btc_63d": -0.5, "sol_vs_btc_63d": -0.6, "bnb_vs_btc_63d": -0.4, "xrp_vs_btc_63d": -0.5}
        assert classify_row(_matrix(raw, z), 0).alt_rotation is AltRotation.BTC_DOMINANCE

    def test_missing_altcoins_degrade_gracefully_not_crash(self):
        raw = {"btc_trend_200d": 0, "btc_mom_63d": 0, "btc_vol_ratio": 1.0}
        call = classify_row(_matrix(raw, {}), 0)
        assert call.alt_rotation is AltRotation.NEUTRAL


class TestEvidence:
    def test_every_call_carries_evidence(self):
        raw = {"btc_trend_200d": 0.1, "btc_mom_63d": 0.1, "btc_vol_ratio": 1.0}
        call = classify_row(_matrix(raw, {}), 0)
        assert len(call.evidence) >= 5
        names = {e.feature for e in call.evidence}
        assert "btc_trend_200d" in names and "btc_vol_ratio" in names

    def test_label_is_trend_and_vol_combined(self):
        raw = {"btc_trend_200d": 0.15, "btc_mom_63d": 0.2, "btc_vol_ratio": 1.5}
        z = {"btc_trend_200d": 1.0, "btc_mom_63d": 1.0}
        call = classify_row(_matrix(raw, z), 0)
        assert call.label == "UP_HIGH"

    def test_incomplete_row_is_flagged_not_hidden(self):
        raw = {"btc_trend_200d": 0.1, "btc_mom_63d": np.nan, "btc_vol_ratio": 1.0}
        call = classify_row(_matrix(raw, {}), 0)
        assert call.complete is False


class TestSerialization:
    def test_to_dict_rounds_floats_for_display(self):
        raw = {"btc_trend_200d": 0.123456789, "btc_mom_63d": 0.1, "btc_vol_ratio": 1.0}
        z = {"btc_trend_200d": 0.987654321}
        d = classify_row(_matrix(raw, z), 0).to_dict()
        trend_ev = next(e for e in d["evidence"] if e["feature"] == "btc_trend_200d")
        assert trend_ev["z"] == pytest.approx(0.988, abs=1e-3)
        assert len(str(trend_ev["z"]).split(".")[-1]) <= 4


class TestLatestAndHistory:
    def test_classify_latest_is_as_of_a_date(self):
        m = _matrix({"btc_trend_200d": 0.1, "btc_mom_63d": 0.1, "btc_vol_ratio": 1.0}, {}, n_rows=5)
        call = classify_latest(m, datetime(2026, 1, 3, tzinfo=UTC))
        assert call.date == "2026-01-03"

    def test_classify_latest_none_before_any_data(self):
        m = _matrix({"btc_trend_200d": 0.1, "btc_mom_63d": 0.1, "btc_vol_ratio": 1.0}, {}, n_rows=5)
        assert classify_latest(m, datetime(2025, 1, 1, tzinfo=UTC)) is None

    def test_classify_history_only_usable_rows(self):
        m = _matrix({"btc_trend_200d": 0.1, "btc_mom_63d": 0.1, "btc_vol_ratio": 1.0}, {"btc_trend_200d": 0.5}, n_rows=10)
        calls = classify_history(m)
        assert len(calls) == len(m.usable_rows())
