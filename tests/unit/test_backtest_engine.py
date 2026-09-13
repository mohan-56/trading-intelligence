"""Backtest engine tests.

The hand-computed equity test is the one that matters most: it would have
caught the real bug found on the first real run -- each loop iteration wrote
both equity[t+1] and equity[t+2], so every index except the last got
overwritten by the NEXT iteration's write before its real price return was
ever used. Buy-and-hold BTC from 2018-2026 showed a LOSS with ~0.1%
annualized vol under that bug; the actual result once fixed was +378% total
return with ~44% vol -- both consistent with BTC's real, well-known history.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from app.core.provenance import Latency, Provenance
from app.domain.backtest.engine import run_backtest
from app.domain.backtest.models import Signal
from app.domain.backtest.risk import RiskEngine
from app.domain.marketdata.models import Bar, BarSeries
from app.domain.symbols.models import AssetClass, Symbol, Timeframe

PROV = Provenance.single("test", Latency.EOD)
SYM = Symbol(ticker="TESTCOIN", asset_class=AssetClass.CRYPTO, exchange="BINANCE")


def _constant_growth_series(n: int, daily_growth: float, start: float = 100.0) -> BarSeries:
    """Open == prior close, every bar identical shape -- makes the expected
    equity trajectory computable by hand with no ambiguity about which price
    field is used where."""
    base = datetime(2024, 1, 1, tzinfo=UTC)
    prices = start * (1.0 + daily_growth) ** np.arange(n)
    bars = [Bar(ts=base + timedelta(days=i), open=float(prices[i]), high=float(prices[i]) * 1.001,
                low=float(prices[i]) * 0.999, close=float(prices[i])) for i in range(n)]
    return BarSeries.from_bars(SYM, Timeframe.D1, bars, PROV)


class AlwaysLong:
    name = "always_long"
    description = "test double"

    def generate_signal(self, view):
        return Signal.LONG


class FixedWeightRisk(RiskEngine):
    """A risk engine stub with a known, constant weight -- removes realized
    volatility from the equation so the equity math is fully predictable."""

    def __init__(self, weight: float) -> None:
        super().__init__()
        self._weight = weight

    def size(self, signal, view):
        return self._weight if signal is Signal.LONG else 0.0


class TestEquityMatchesHandComputedCompounding:
    def test_constant_growth_at_full_weight_matches_hand_computed_equity(self):
        """This is the regression test for the double-write bug. Weight is
        fixed at 1.0, growth is a known constant rate, zero cost -- the
        entire equity path is exactly (1+g)^k for k steps, nothing else."""
        n, g = 120, 0.01
        price = _constant_growth_series(n, g)
        result = run_backtest(AlwaysLong(), price, risk_engine=FixedWeightRisk(1.0), cost_bps=0.0, warmup_days=20)

        equity = np.array(result.equity_curve)
        # Independent reference: a plain manual loop, not an algebraic
        # closed form -- the closed form is exactly where an earlier draft
        # of this test made an off-by-one deriving the exponent by hand.
        # equity_curve[0], [1] are the pre-trade seed (still 1.0); every
        # index from [2] onward compounds by exactly (1+g).
        expected = 1.0
        for _ in range(len(equity) - 2):
            expected *= 1.0 + g
        assert equity[-1] == pytest.approx(expected, rel=1e-9)

    def test_negative_growth_produces_a_loss_not_a_flat_line(self):
        """The bug's symptom: a real, sustained price move collapsed to
        near-1.0 flat equity. A falling price at full weight must show up
        as an actual loss."""
        n, g = 100, -0.01
        price = _constant_growth_series(n, g)
        result = run_backtest(AlwaysLong(), price, risk_engine=FixedWeightRisk(1.0), cost_bps=0.0, warmup_days=20)
        assert result.equity_curve[-1] < 0.5  # a sustained -1%/day move over ~80 steps is a large loss

    def test_zero_weight_never_moves_equity_regardless_of_price(self):
        n = 100
        price = _constant_growth_series(n, 0.05)  # aggressive growth, but strategy stays flat
        result = run_backtest(AlwaysLong(), price, risk_engine=FixedWeightRisk(0.0), cost_bps=10.0, warmup_days=20)
        assert result.equity_curve[-1] == pytest.approx(1.0, abs=1e-9)


class TestFillTiming:
    def test_signal_decided_at_t_fills_at_t_plus_1_open_not_t_close(self):
        """A strategy that goes LONG only once, triggered at a specific bar,
        must not earn any return from that bar's own close -- only from
        open[t+1] onward. Construct a huge one-day jump exactly at the
        trigger bar's close and confirm it is NOT captured."""
        n = 60
        base = datetime(2024, 1, 1, tzinfo=UTC)
        trigger_t = 30
        # A huge spike in bar `trigger_t`'s own close, invisible at open --
        # if the engine let the strategy "trade" that bar's own close, this
        # spike would inflate the entry.
        bars = [Bar(ts=base + timedelta(days=i), open=100.0, high=105.0, low=99.0,
                    close=500.0 if i == trigger_t else 100.0) for i in range(n)]
        price = BarSeries.from_bars(SYM, Timeframe.D1, bars, PROV)

        class TriggerOnce:
            name = "trigger_once"
            description = "test double"

            def generate_signal(self, view):
                return Signal.LONG if len(view.price) - 1 >= trigger_t else Signal.FLAT

        result = run_backtest(TriggerOnce(), price, risk_engine=FixedWeightRisk(1.0), cost_bps=0.0, warmup_days=10)
        opens = [b.open for b in bars]
        assert all(o == pytest.approx(100.0) for o in opens)  # every open is 100 -- the spike never touched it
        # equity must never reflect the 500.0 spike at all, since open never moved
        assert max(result.equity_curve) < 1.5


class TestCosts:
    def test_cost_charged_on_entry_reduces_equity_by_expected_amount(self):
        n = 60
        price = _constant_growth_series(n, 0.0)  # flat price: isolates the cost effect
        cost_bps = 50.0  # 0.5%
        result = run_backtest(AlwaysLong(), price, risk_engine=FixedWeightRisk(1.0), cost_bps=cost_bps, warmup_days=20)
        # one entry, turnover 1.0 -> cost = 1.0 * 50/10000 = 0.5%
        assert result.equity_curve[-1] == pytest.approx(1.0 - 0.005, abs=1e-6)

    def test_zero_cost_and_zero_growth_is_exactly_flat(self):
        price = _constant_growth_series(60, 0.0)
        result = run_backtest(AlwaysLong(), price, risk_engine=FixedWeightRisk(1.0), cost_bps=0.0, warmup_days=20)
        assert result.equity_curve[-1] == pytest.approx(1.0, abs=1e-9)


class TestTradeBookkeeping:
    def test_going_flat_closes_the_open_trade_with_correct_pnl(self):
        n = 80
        base = datetime(2024, 1, 1, tzinfo=UTC)
        flip_at = 50
        bars = [Bar(ts=base + timedelta(days=i), open=100.0 * (1.02 ** min(i, flip_at)),
                    high=1, low=1, close=1) for i in range(n)]
        # fix high/low to bracket open sanely
        bars = [Bar(ts=b.ts, open=b.open, high=b.open * 1.001, low=b.open * 0.999, close=b.open) for b in bars]
        price = BarSeries.from_bars(SYM, Timeframe.D1, bars, PROV)

        class FlipOnce:
            name = "flip_once"
            description = "test double"

            def generate_signal(self, view):
                return Signal.LONG if len(view.price) - 1 < flip_at else Signal.FLAT

        result = run_backtest(FlipOnce(), price, risk_engine=FixedWeightRisk(1.0), cost_bps=0.0, warmup_days=10)
        closed = [t for t in result.trades if t.pnl_pct is not None]
        assert len(closed) == 1
        assert closed[0].pnl_pct > 0  # price rose the whole time it was held

    def test_still_open_at_end_reports_none_pnl_not_fabricated(self):
        price = _constant_growth_series(60, 0.01)
        result = run_backtest(AlwaysLong(), price, risk_engine=FixedWeightRisk(1.0), cost_bps=0.0, warmup_days=20)
        assert len(result.trades) == 1
        assert result.trades[0].exit_date is None
        assert result.trades[0].pnl_pct is None


class TestQuarantine:
    def test_a_broken_bar_is_quarantined_not_crashed_on(self):
        n = 60
        base = datetime(2024, 1, 1, tzinfo=UTC)
        bars = [Bar(ts=base + timedelta(days=i), open=100.0, high=101.0, low=99.0, close=100.0) for i in range(n)]
        # Corrupt one bar's open to NaN, deep enough to be hit by the loop.
        bars[40] = Bar(ts=bars[40].ts, open=float("nan"), high=101.0, low=99.0, close=100.0)
        price = BarSeries.from_bars(SYM, Timeframe.D1, bars, PROV)
        result = run_backtest(AlwaysLong(), price, risk_engine=FixedWeightRisk(1.0), cost_bps=0.0, warmup_days=10)
        assert result.quarantined_days >= 1


class TestReproducibility:
    def test_identical_spec_produces_identical_result(self):
        price = _constant_growth_series(100, 0.01)
        r1 = run_backtest(AlwaysLong(), price, risk_engine=FixedWeightRisk(0.5), cost_bps=10.0, warmup_days=20)
        r2 = run_backtest(AlwaysLong(), price, risk_engine=FixedWeightRisk(0.5), cost_bps=10.0, warmup_days=20)
        assert r1.equity_curve == r2.equity_curve
        assert r1.metrics == r2.metrics
        assert r1.spec == r2.spec
