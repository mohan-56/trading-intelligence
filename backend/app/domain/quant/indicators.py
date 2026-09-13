"""Technical indicators -- pure, vectorized, testable.

Every function here obeys three rules:

1. **Causal.** Output at index i uses only inputs at index <= i. No centred
   windows, no full-sample normalization. An indicator that peeks is a
   look-ahead bug that makes every downstream backtest look brilliant and
   be wrong.
2. **Aligned.** Output is always the same length as the input, NaN-padded at
   the front. Dropping the warm-up period silently shifts the series by
   `window` bars, which is the same bug wearing a disguise.
3. **Documented with its failure mode.** Reading these should teach you when
   each one lies to you, not just what it computes.
"""

from __future__ import annotations

import numpy as np

TRADING_DAYS = 365  # crypto trades every day of the year, unlike equities


def _validate(values: np.ndarray, window: int) -> np.ndarray:
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    return np.asarray(values, dtype=np.float64)


def sma(values: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average."""
    v = _validate(values, window)
    out = np.full(len(v), np.nan)
    if len(v) < window:
        return out
    cumsum = np.cumsum(np.insert(v, 0, 0.0))
    out[window - 1:] = (cumsum[window:] - cumsum[:-window]) / window
    return out


def rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    v = _validate(values, window)
    out = np.full(len(v), np.nan)
    if len(v) < window:
        return out
    windows = np.lib.stride_tricks.sliding_window_view(v, window)
    out[window - 1:] = np.std(windows, axis=1)
    return out


def realized_vol(values: np.ndarray, window: int = 21, annualize: bool = True) -> np.ndarray:
    """Annualized realized volatility from log returns.

    Backward-looking, unlike an options-implied vol -- crypto has no free
    VIX-equivalent, so this is the only volatility signal this product has
    for the crypto legs themselves.
    """
    v = _validate(values, window)
    returns = np.full(len(v), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        returns[1:] = np.log(v[1:] / v[:-1])
    vol = rolling_std(returns, window)
    return vol * np.sqrt(TRADING_DAYS) if annualize else vol


def vol_ratio(values: np.ndarray, window: int = 21, slow_window: int = 90) -> np.ndarray:
    """Short-horizon vol / long-horizon vol. >1 = expanding, <1 = compressing.

    Slow window is 90 days here, not the ~252 a traditional-market version
    would use -- crypto vol regimes rotate faster, and a full-year baseline
    is stale by the time a compression actually resolves.
    """
    fast = realized_vol(values, window)
    slow = realized_vol(values, slow_window)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(slow > 0, fast / slow, np.nan)


def pct_from_sma(values: np.ndarray, window: int) -> np.ndarray:
    """Percent distance from the moving average -- scale-free trend measure,
    comparable across BTC at $77k and a token at $0.001 without adjustment."""
    v = _validate(values, window)
    ma = sma(v, window)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (v - ma) / ma


def log_return(values: np.ndarray, window: int = 1) -> np.ndarray:
    v = _validate(values, window)
    out = np.full(len(v), np.nan)
    if len(v) <= window:
        return out
    with np.errstate(divide="ignore", invalid="ignore"):
        out[window:] = np.log(v[window:] / v[:-window])
    return out


def diff(values: np.ndarray, window: int = 1) -> np.ndarray:
    """Absolute change -- use for things already in percent (a yield, a
    funding rate). A 10y yield going 2%->3% is +100bp; calling it '+50%'
    is meaningless."""
    v = _validate(values, window)
    out = np.full(len(v), np.nan)
    if len(v) > window:
        out[window:] = v[window:] - v[:-window]
    return out


def relative_strength(a: np.ndarray, b: np.ndarray, window: int) -> np.ndarray:
    """Return of A minus return of B -- is A beating B? Positive = A leading.
    ETH vs BTC over a rolling window is the standard 'alt season' read."""
    return log_return(a, window) - log_return(b, window)


def expanding_zscore(values: np.ndarray, min_periods: int = 60) -> np.ndarray:
    """Z-score against ALL history up to t, never a fixed window and never
    full-sample. This is what the regime engine standardizes with: on any
    given day it must know nothing about days after it."""
    v = np.asarray(values, dtype=np.float64)
    out = np.full(len(v), np.nan)
    if len(v) < min_periods:
        return out
    valid = ~np.isnan(v)
    filled = np.where(valid, v, 0.0)
    counts = np.cumsum(valid)
    sums = np.cumsum(filled)
    sumsq = np.cumsum(filled ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = sums / counts
        var = np.maximum(sumsq / counts - mean ** 2, 0.0)
        sd = np.sqrt(var)
        z = (v - mean) / sd
    out[counts >= min_periods] = z[counts >= min_periods]
    return np.where(np.isfinite(out), out, np.nan)


def drawdown(values: np.ndarray) -> np.ndarray:
    """Percent below the running peak. Always <= 0."""
    v = np.asarray(values, dtype=np.float64)
    peak = np.maximum.accumulate(np.where(np.isnan(v), -np.inf, v))
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = (v - peak) / peak
    return np.where(np.isfinite(dd), dd, np.nan)
