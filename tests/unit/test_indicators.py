"""Indicator tests. Hand-computed expected values -- checking one
implementation against another library only proves they share a bug.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.domain.quant import indicators as ind


class TestSMA:
    def test_hand_computed(self):
        v = np.array([1.0, 2, 3, 4, 5])
        out = ind.sma(v, 3)
        assert np.isnan(out[:2]).all()
        assert out[2] == pytest.approx(2.0)
        assert out[4] == pytest.approx(4.0)

    def test_too_short_is_all_nan(self):
        assert np.isnan(ind.sma(np.array([1.0, 2.0]), 5)).all()


class TestVolatility:
    def test_rolling_std_hand_computed(self):
        v = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        assert ind.rolling_std(v, 8)[-1] == pytest.approx(2.0)

    def test_realized_vol_recovers_known_sigma(self):
        rng = np.random.default_rng(4)
        daily_sigma = 0.02
        prices = 100 * np.exp(np.cumsum(rng.normal(0, daily_sigma, 2000)))
        got = np.nanmean(ind.realized_vol(prices, 252))
        assert got == pytest.approx(daily_sigma * np.sqrt(365), rel=0.15)

    def test_vol_ratio_above_one_when_vol_expands(self):
        rng = np.random.default_rng(8)
        calm = rng.normal(0, 0.006, 400)
        stormy = rng.normal(0, 0.05, 30)
        prices = 100 * np.exp(np.cumsum(np.concatenate([calm, stormy])))
        assert ind.vol_ratio(prices, 21, 90)[-1] > 1.0


class TestTransforms:
    def test_pct_from_sma_zero_on_flat_series(self):
        assert ind.pct_from_sma(np.full(50, 100.0), 20)[-1] == pytest.approx(0.0)

    def test_log_return_hand_computed(self):
        assert ind.log_return(np.array([100.0, 110.0]), 1)[1] == pytest.approx(np.log(1.1))

    def test_log_returns_add_across_time(self):
        v = np.array([100.0, 110.0, 121.0])
        r = ind.log_return(v, 1)
        assert r[1] + r[2] == pytest.approx(ind.log_return(v, 2)[2])

    def test_diff_is_absolute_not_relative(self):
        assert ind.diff(np.array([2.0, 3.0]), 1)[1] == pytest.approx(1.0)

    def test_relative_strength_sign(self):
        a = np.array([100.0] * 10 + [120.0])
        b = np.array([100.0] * 10 + [110.0])
        assert ind.relative_strength(a, b, 10)[-1] > 0

    def test_drawdown_never_positive(self):
        rng = np.random.default_rng(9)
        v = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 500)))
        dd = ind.drawdown(v)
        assert (dd[~np.isnan(dd)] <= 1e-12).all()

    def test_drawdown_hand_computed(self):
        dd = ind.drawdown(np.array([100.0, 120.0, 60.0, 90.0]))
        assert dd[1] == pytest.approx(0.0)
        assert dd[2] == pytest.approx(-0.5)


class TestCausality:
    """The tests that matter most. An indicator that sees the future turns
    every downstream regime call and backtest into fiction."""

    @pytest.mark.parametrize("fn", [
        lambda v: ind.sma(v, 10),
        lambda v: ind.rolling_std(v, 10),
        lambda v: ind.realized_vol(v, 21),
        lambda v: ind.pct_from_sma(v, 20),
        lambda v: ind.log_return(v, 5),
        lambda v: ind.drawdown(v),
        lambda v: ind.vol_ratio(v, 21, 60),
    ])
    def test_future_data_cannot_change_the_past(self, fn):
        rng = np.random.default_rng(11)
        full = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 400)))
        cut = 300
        a = fn(full)[:cut]
        b = fn(full[:cut].copy())
        both = ~np.isnan(a) & ~np.isnan(b)
        assert both.sum() > 50
        np.testing.assert_allclose(a[both], b[both], rtol=1e-9, atol=1e-9)

    def test_expanding_zscore_is_causal(self):
        rng = np.random.default_rng(12)
        v = rng.normal(0, 1, 800)
        cut = 500
        a = ind.expanding_zscore(v, 60)[:cut]
        b = ind.expanding_zscore(v[:cut].copy(), 60)
        both = ~np.isnan(a) & ~np.isnan(b)
        np.testing.assert_allclose(a[both], b[both], rtol=1e-8, atol=1e-8)

    def test_expanding_zscore_normalizes(self):
        rng = np.random.default_rng(13)
        z = ind.expanding_zscore(rng.normal(50, 10, 2000), 60)
        valid = z[~np.isnan(z)]
        assert abs(np.mean(valid)) < 0.2
        assert 0.7 < np.std(valid) < 1.4
