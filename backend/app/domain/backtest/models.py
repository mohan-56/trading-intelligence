"""Backtest domain models.

`MarketView` is the safety primitive everything else depends on. A strategy
never receives a DataFrame or a raw BarSeries with the full timeline -- it
receives a MarketView whose every series has already been clipped to `now`
by the engine, via `BarSeries.slice_until()`. There is no method on
MarketView that returns anything past `now`. Look-ahead bias isn't
discouraged by convention here; it's structurally inexpressible -- a
strategy would have to reach around the object to get at future data, not
just forget a guard clause.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from app.domain.marketdata.models import BarSeries


class Signal(StrEnum):
    """LONG / FLAT only for now -- this is a spot-only backtester. A SHORT
    signal would imply borrow costs, margin, and liquidation mechanics this
    engine doesn't model yet; adding it later means implementing those, not
    just allowing a negative weight and pretending it's the same thing."""

    LONG = "LONG"
    FLAT = "FLAT"


@dataclass(frozen=True, slots=True)
class MarketView:
    """What a strategy is allowed to see. `series` values are already
    clipped to `now` -- e.g. {"price": BarSeries via slice_until(now),
    "funding": ...}. A strategy that wants funding-rate data asks for it by
    name; whatever it asks for, it's already safe."""

    now: datetime
    series: dict[str, BarSeries]

    @property
    def price(self) -> BarSeries:
        return self.series["price"]

    def get(self, key: str) -> BarSeries | None:
        return self.series.get(key)

    def has(self, key: str) -> bool:
        s = self.series.get(key)
        return s is not None and s.is_available


@dataclass(frozen=True, slots=True)
class Trade:
    """One contiguous holding period at a nonzero weight -- opens when weight
    moves away from 0, closes when it returns to 0 (or at the end of the
    backtest, marked open)."""

    entry_date: str
    exit_date: str | None
    entry_price: float
    exit_price: float | None
    weight: float
    pnl_pct: float | None  # None while still open

    def to_dict(self) -> dict:
        return {
            "entry_date": self.entry_date, "exit_date": self.exit_date,
            "entry_price": round(self.entry_price, 6), "exit_price": round(self.exit_price, 6) if self.exit_price else None,
            "weight": round(self.weight, 4), "pnl_pct": round(self.pnl_pct, 4) if self.pnl_pct is not None else None,
        }


@dataclass(frozen=True, slots=True)
class BacktestSpec:
    """Content-hashable record of exactly what produced a result -- the
    reproducibility contract. Two runs with an identical spec must produce
    an identical result, byte for byte."""

    strategy_name: str
    symbol: str
    start: str
    end: str
    cost_bps: float
    target_vol_annual: float
    max_weight: float
    warmup_days: int

    def to_dict(self) -> dict:
        return {
            "strategy_name": self.strategy_name, "symbol": self.symbol, "start": self.start, "end": self.end,
            "cost_bps": self.cost_bps, "target_vol_annual": self.target_vol_annual,
            "max_weight": self.max_weight, "warmup_days": self.warmup_days,
        }


@dataclass(frozen=True)
class BacktestResult:
    spec: BacktestSpec
    dates: list[str]
    equity_curve: list[float]  # starts at 1.0
    weights: list[float]       # target weight held into each date
    trades: tuple[Trade, ...]
    metrics: dict
    quarantined_days: int = field(default=0)  # days the strategy was skipped (missing data), not silently zero-filled

    def to_dict(self, max_points: int = 2000) -> dict:
        n = len(self.dates)
        idx = list(range(n))
        if n > max_points:
            import numpy as np
            idx = sorted(set(np.linspace(0, n - 1, max_points).astype(int).tolist()))
        return {
            "spec": self.spec.to_dict(),
            "metrics": self.metrics,
            "n_trades": len(self.trades),
            "quarantined_days": self.quarantined_days,
            "trades": [t.to_dict() for t in self.trades],
            "equity_curve": [{"date": self.dates[i], "equity": round(self.equity_curve[i], 6), "weight": round(self.weights[i], 4)} for i in idx],
        }
