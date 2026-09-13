"""Strategies. Each is a pure function of a MarketView -> Signal. No strategy
sizes its own position -- that's the RiskEngine's job, deliberately kept
separate (see risk.py).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from app.domain.backtest.models import MarketView, Signal
from app.domain.quant.indicators import log_return, rolling_zscore, sma


class Strategy(Protocol):
    name: str
    description: str

    def generate_signal(self, view: MarketView) -> Signal: ...


class BuyAndHold:
    """The baseline every other strategy has to actually beat. A strategy
    that can't outperform this on a risk-adjusted basis isn't earning its
    complexity -- comparing against a trivial baseline is not optional."""

    name = "buy_and_hold"
    description = "Always long. The number every other strategy must beat."

    def generate_signal(self, view: MarketView) -> Signal:
        return Signal.LONG


class Momentum:
    """Long when trailing N-day momentum is positive, flat otherwise.
    Trend-following: works when regimes persist, whipsaws in a chop."""

    name = "momentum_63d"
    description = "Long when the 63-day return is positive, flat otherwise."

    def __init__(self, lookback: int = 63) -> None:
        self.lookback = lookback
        self.name = f"momentum_{lookback}d"

    def generate_signal(self, view: MarketView) -> Signal:
        close = view.price.close
        if len(close) < self.lookback + 1:
            return Signal.FLAT
        mom = log_return(close, self.lookback)[-1]
        return Signal.LONG if np.isfinite(mom) and mom > 0 else Signal.FLAT


class MeanReversion:
    """Long after a sharp short-term drop (fade the move), flat otherwise.
    The mirror image of momentum: works in chop, gets run over in a real
    trend -- which is exactly why comparing both against the same baseline,
    across the same regime history, is the point of building this engine."""

    name = "mean_reversion_10d"
    description = "Long when the 10-day return is more than 1.5 rolling-sigma below its own mean, flat otherwise."

    def __init__(self, lookback: int = 10, z_threshold: float = -1.5, z_window: int = 90) -> None:
        self.lookback = lookback
        self.z_threshold = z_threshold
        self.z_window = z_window
        self.name = f"mean_reversion_{lookback}d"

    def generate_signal(self, view: MarketView) -> Signal:
        close = view.price.close
        if len(close) < self.z_window + self.lookback + 1:
            return Signal.FLAT
        short_return = log_return(close, self.lookback)
        z = rolling_zscore(short_return, self.z_window)[-1]
        return Signal.LONG if np.isfinite(z) and z < self.z_threshold else Signal.FLAT


class FundingContrarian:
    """Crypto-native: fade extreme perpetual funding. Very negative funding
    means shorts are paying longs -- crowded short positioning, which has
    historically preceded squeezes higher. This is a real, commonly cited
    heuristic; it is not backed here by anything beyond what the backtest
    itself shows, which is the entire point of running it rather than
    trusting the heuristic on faith.

    Needs a 'funding' series on the MarketView -- flat if it isn't present,
    never silently substitutes price-only logic for a funding-based one.
    """

    name = "funding_contrarian"
    description = "Long when trailing funding rate is unusually negative (crowded shorts), flat otherwise."

    def __init__(self, z_threshold: float = -1.0, avg_window: int = 21, z_window: int = 180) -> None:
        self.z_threshold = z_threshold
        self.avg_window = avg_window
        self.z_window = z_window

    def generate_signal(self, view: MarketView) -> Signal:
        if not view.has("funding"):
            return Signal.FLAT
        funding = view.get("funding").close
        if len(funding) < self.z_window + self.avg_window + 1:
            return Signal.FLAT
        avg_funding = sma(funding, self.avg_window)
        z = rolling_zscore(avg_funding, self.z_window)[-1]
        return Signal.LONG if np.isfinite(z) and z < self.z_threshold else Signal.FLAT


def default_strategies() -> list[Strategy]:
    return [BuyAndHold(), Momentum(), MeanReversion(), FundingContrarian()]
