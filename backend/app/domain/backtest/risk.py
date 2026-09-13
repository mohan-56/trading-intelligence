"""Risk engine -- independent of signal generation, on purpose.

A strategy decides direction (LONG or FLAT). It never decides size. That
split exists so a strategy that goes "all in" whenever it has an opinion
can't actually blow up the portfolio -- position size is a function of
realized volatility, not conviction, and it's computed here, in one place,
the same way for every strategy this backtester ever runs.
"""

from __future__ import annotations

import numpy as np

from app.domain.backtest.models import MarketView, Signal
from app.domain.quant.indicators import realized_vol


class RiskEngine:
    def __init__(self, target_vol_annual: float = 0.40, max_weight: float = 1.0, vol_lookback: int = 21) -> None:
        """target_vol_annual: the portfolio volatility this sizing aims for.
        0.40 (40% annualized) is a deliberately modest target for crypto,
        which routinely realizes 60-100%+ annualized vol on BTC alone --
        sizing to match crypto's own volatility would mean routinely being
        near-fully levered into the asset with the least room for error.

        max_weight: hard cap regardless of how calm volatility looks. Low
        realized vol is not a promise of continued low vol -- it is
        frequently the precursor to the opposite.
        """
        self.target_vol_annual = target_vol_annual
        self.max_weight = max_weight
        self.vol_lookback = vol_lookback

    def size(self, signal: Signal, view: MarketView) -> float:
        """Returns a target weight in [0, max_weight]. FLAT is always 0,
        regardless of volatility -- there is no such thing as a risk-managed
        position in an asset you have no view on."""
        if signal is not Signal.FLAT and signal is not Signal.LONG:
            raise ValueError(f"unknown signal: {signal}")
        if signal is Signal.FLAT:
            return 0.0

        close = view.price.close
        if len(close) < self.vol_lookback + 1:
            return 0.0  # not enough history to size safely -- stay flat, don't guess

        # realized_vol is a trailing-window function -- its LAST value depends
        # only on the last (vol_lookback+1) closes, never on anything before
        # that. Computing it over the full, ever-growing `close` array (which
        # is what MarketView hands us -- everything up to `now`) recomputes
        # the whole rolling-std series just to read its final entry: O(n) of
        # wasted work at every single bar, O(n^2) over a full backtest.
        # Measured on a real 3,113-bar BTC backtest: 1.56s of a 1.98s total
        # was inside this one call before slicing the tail. Truncating to
        # the tail first gives an identical value, computed once, not n times.
        vol = realized_vol(close[-(self.vol_lookback + 1):], self.vol_lookback)[-1]
        if not np.isfinite(vol) or vol <= 0:
            return 0.0  # a broken or zero vol reading must never size a position

        raw_weight = self.target_vol_annual / vol
        return float(min(raw_weight, self.max_weight))
