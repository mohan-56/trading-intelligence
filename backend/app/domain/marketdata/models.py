"""Market data models. Columnar (numpy), not list-of-objects -- ~10x smaller
in RAM and what the quant layer wants anyway.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from app.core.provenance import Provenance
from app.domain.symbols.models import Symbol, Timeframe

OHLCV_FIELDS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True, slots=True)
class Bar:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    def is_coherent(self) -> bool:
        if not all(np.isfinite([self.open, self.high, self.low, self.close])):
            return False
        if self.high < self.low:
            return False
        if not (self.low <= self.open <= self.high):
            return False
        if not (self.low <= self.close <= self.high):
            return False
        return self.volume >= 0


@dataclass(frozen=True, eq=False)
class BarSeries:
    symbol: Symbol
    timeframe: Timeframe
    ts: np.ndarray  # datetime64[s], strictly increasing
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    provenance: Provenance

    def __post_init__(self) -> None:
        n = len(self.ts)
        for f in OHLCV_FIELDS:
            if len(getattr(self, f)) != n:
                raise ValueError(f"BarSeries column length mismatch: {f}")

    @classmethod
    def empty(cls, symbol: Symbol, timeframe: Timeframe, provenance: Provenance) -> BarSeries:
        z = np.array([], dtype=np.float64)
        return cls(symbol=symbol, timeframe=timeframe, ts=np.array([], dtype="datetime64[s]"),
                   open=z, high=z, low=z, close=z, volume=z, provenance=provenance)

    @classmethod
    def unavailable(cls, symbol: Symbol, timeframe: Timeframe, reason: str) -> BarSeries:
        return cls.empty(symbol, timeframe, Provenance.unavailable(reason))

    @classmethod
    def from_bars(cls, symbol: Symbol, timeframe: Timeframe, bars: list[Bar], provenance: Provenance) -> BarSeries:
        if not bars:
            return cls.empty(symbol, timeframe, provenance)
        bars = sorted(bars, key=lambda b: b.ts)
        return cls(
            symbol=symbol, timeframe=timeframe,
            ts=np.array([np.datetime64(b.ts.replace(tzinfo=None), "s") for b in bars]),
            open=np.array([b.open for b in bars], dtype=np.float64),
            high=np.array([b.high for b in bars], dtype=np.float64),
            low=np.array([b.low for b in bars], dtype=np.float64),
            close=np.array([b.close for b in bars], dtype=np.float64),
            volume=np.array([b.volume for b in bars], dtype=np.float64),
            provenance=provenance,
        )

    def __len__(self) -> int:
        return len(self.ts)

    @property
    def is_empty(self) -> bool:
        return len(self.ts) == 0

    @property
    def is_available(self) -> bool:
        return not self.is_empty and self.provenance.latency.is_usable

    @property
    def start(self) -> datetime | None:
        return None if self.is_empty else _to_dt(self.ts[0])

    @property
    def end(self) -> datetime | None:
        return None if self.is_empty else _to_dt(self.ts[-1])

    def _rebuild(self, mask_or_slice, provenance: Provenance | None = None) -> BarSeries:
        return BarSeries(
            symbol=self.symbol, timeframe=self.timeframe, ts=self.ts[mask_or_slice],
            open=self.open[mask_or_slice], high=self.high[mask_or_slice],
            low=self.low[mask_or_slice], close=self.close[mask_or_slice], volume=self.volume[mask_or_slice],
            provenance=provenance or self.provenance,
        )

    def slice_until(self, t: datetime) -> BarSeries:
        """Bars at or before `t` -- the primitive that makes a backtester's
        MarketView physically unable to see the future."""
        cut = np.datetime64(t.replace(tzinfo=None), "s")
        return self._rebuild(self.ts <= cut)

    def slice_range(self, start: datetime | None, end: datetime | None) -> BarSeries:
        mask = np.ones(len(self.ts), dtype=bool)
        if start is not None:
            mask &= self.ts >= np.datetime64(start.replace(tzinfo=None), "s")
        if end is not None:
            mask &= self.ts <= np.datetime64(end.replace(tzinfo=None), "s")
        return self._rebuild(mask)

    def tail(self, n: int) -> BarSeries:
        return self._rebuild(slice(-n, None)) if n < len(self) else self

    def log_returns(self) -> np.ndarray:
        if len(self) < 2:
            return np.full(len(self), np.nan)
        out = np.full(len(self), np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[1:] = np.log(self.close[1:] / self.close[:-1])
        return out

    def to_records(self, max_points: int | None = None) -> list[dict]:
        idx = np.arange(len(self))
        if max_points and len(self) > max_points:
            idx = np.unique(np.linspace(0, len(self) - 1, max_points).astype(int))
        return [
            {"ts": _to_dt(self.ts[i]).isoformat(), "open": float(self.open[i]), "high": float(self.high[i]),
             "low": float(self.low[i]), "close": float(self.close[i]), "volume": float(self.volume[i])}
            for i in idx
        ]

    def describe(self) -> dict:
        return {
            "symbol": self.symbol.ticker, "timeframe": self.timeframe.value, "rows": len(self),
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
            "provenance": self.provenance.to_dict(),
        }


def _to_dt(value: np.datetime64) -> datetime:
    return value.astype("datetime64[s]").astype(datetime).replace(tzinfo=UTC)
