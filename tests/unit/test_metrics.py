"""Backtest metrics -- hand-computed, never checked against another library."""

from __future__ import annotations

import numpy as np
import pytest

from app.domain.backtest import metrics as m


class TestReturnsAndTotalReturn:
    def test_daily_returns_hand_computed(self):
        eq = np.array([1.0, 1.1, 1.21])
        r = m.daily_returns(eq)
        assert np.isnan(r[0])
        assert r[1] == pytest.approx(0.1)
        assert r[2] == pytest.approx(0.1)

    def test_total_return_hand_computed(self):
        assert m.total_return(np.array([1.0, 1.5])) == pytest.approx(0.5)

    def test_total_return_empty_is_zero(self):
        assert m.total_return(np.array([])) == 0.0


class TestCAGR:
    def test_doubling_in_one_year(self):
        # 365 days at 365 trading days/year = exactly 1 year
        eq = np.array([1.0, 2.0])
        assert m.cagr(eq, n_days=365) == pytest.approx(1.0, abs=1e-6)

    def test_doubling_in_two_years_is_sqrt2_minus_1(self):
        eq = np.array([1.0, 2.0])
        assert m.cagr(eq, n_days=730) == pytest.approx(np.sqrt(2) - 1, abs=1e-6)

    def test_zero_days_is_zero_not_a_crash(self):
        assert m.cagr(np.array([1.0, 2.0]), n_days=0) == 0.0


class TestSharpeAndSortino:
    def test_zero_vol_is_zero_not_infinite(self):
        r = np.array([np.nan, 0.001, 0.001, 0.001, 0.001])
        assert m.sharpe_ratio(r) == 0.0

    def test_positive_drift_gives_positive_sharpe(self):
        rng = np.random.default_rng(1)
        r = rng.normal(0.002, 0.01, 500)
        assert m.sharpe_ratio(r) > 0

    def test_sortino_ignores_upside_volatility(self):
        """Two series with identical downside AND identical upside MEAN, but
        very different upside DISPERSION, must show a far smaller Sortino
        gap than Sharpe gap -- Sharpe penalizes the extra upside variance,
        Sortino (downside-only) never sees it. An earlier version of this
        test varied the upside mean as well as its spread, which changes
        both metrics for an unrelated reason and made the assertion fail
        for a reason that had nothing to do with Sortino's actual property."""
        base_down = [-0.02, -0.01, -0.03, -0.01, -0.02] * 20
        calm_up = [0.01] * 100                               # mean 0.01, std 0
        wild_up = [0.0, 0.02, 0.0, 0.02, 0.01] * 20           # mean 0.01, std > 0
        assert np.mean(calm_up) == pytest.approx(np.mean(wild_up))  # the one thing held equal
        r1 = np.array(base_down + calm_up)
        r2 = np.array(base_down + wild_up)
        sharpe_gap = abs(m.sharpe_ratio(r1) - m.sharpe_ratio(r2))
        sortino_gap = abs(m.sortino_ratio(r1) - m.sortino_ratio(r2))
        assert sortino_gap < sharpe_gap


class TestMaxDrawdown:
    def test_hand_computed(self):
        eq = np.array([1.0, 1.2, 0.6, 0.9])
        assert m.max_drawdown(eq) == pytest.approx(-0.5)  # 0.6 vs peak 1.2

    def test_monotonic_up_has_zero_drawdown(self):
        assert m.max_drawdown(np.array([1.0, 1.1, 1.3, 1.5])) == pytest.approx(0.0)

    def test_never_positive(self):
        rng = np.random.default_rng(2)
        eq = np.cumprod(1 + rng.normal(0.001, 0.02, 500))
        assert m.max_drawdown(eq) <= 1e-12


class TestWinRateAndProfitFactor:
    def test_win_rate_hand_computed(self):
        assert m.win_rate([0.1, -0.05, 0.2, -0.1]) == pytest.approx(0.5)

    def test_win_rate_ignores_open_trades(self):
        assert m.win_rate([0.1, None, 0.2]) == pytest.approx(1.0)

    def test_profit_factor_hand_computed(self):
        # gains = 0.3, losses = 0.15 -> 2.0
        assert m.profit_factor([0.1, 0.2, -0.1, -0.05]) == pytest.approx(2.0)

    def test_profit_factor_no_losses_is_infinite(self):
        assert m.profit_factor([0.1, 0.2]) == float("inf")

    def test_profit_factor_no_trades_is_zero(self):
        assert m.profit_factor([]) == 0.0


class TestAllMetrics:
    def test_returns_every_expected_key(self):
        eq = np.array([1.0, 1.05, 1.02, 1.1])
        result = m.all_metrics(eq, n_days=100, trade_pnls=[0.05, -0.02])
        for key in ("total_return", "cagr", "annualized_vol", "sharpe", "sortino",
                    "max_drawdown", "calmar", "win_rate", "profit_factor", "n_trades"):
            assert key in result
