"""Backtest performance metrics -- pure functions over an equity curve.

Every one is testable against a hand-computed value, not just "runs without
crashing." A Sharpe ratio nobody has verified by hand is a Sharpe ratio
nobody should trust.
"""

from __future__ import annotations

import numpy as np

TRADING_DAYS = 365  # crypto trades every day of the year


def daily_returns(equity_curve: np.ndarray) -> np.ndarray:
    out = np.full(len(equity_curve), np.nan)
    if len(equity_curve) > 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            out[1:] = equity_curve[1:] / equity_curve[:-1] - 1.0
    return out


def total_return(equity_curve: np.ndarray) -> float:
    if len(equity_curve) < 1 or equity_curve[0] == 0:
        return 0.0
    return float(equity_curve[-1] / equity_curve[0] - 1.0)


def cagr(equity_curve: np.ndarray, n_days: int) -> float:
    """Compound annual growth rate. n_days, not len(equity_curve) - 1 --
    callers pass the actual elapsed calendar days, since a quarantined-day
    gap would otherwise understate the holding period."""
    if n_days <= 0 or equity_curve[0] <= 0 or equity_curve[-1] <= 0:
        return 0.0
    years = n_days / TRADING_DAYS
    if years <= 0:
        return 0.0
    return float((equity_curve[-1] / equity_curve[0]) ** (1.0 / years) - 1.0)


def annualized_vol(returns: np.ndarray) -> float:
    clean = returns[np.isfinite(returns)]
    if len(clean) < 2:
        return 0.0
    return float(np.std(clean, ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: np.ndarray, risk_free_annual: float = 0.0) -> float:
    """Annualized Sharpe. risk_free_annual defaults to 0 -- deliberately: a
    real risk-free comparison needs a real rate series, and pretending 0% is
    'conservative' is itself a claim worth stating rather than assuming."""
    clean = returns[np.isfinite(returns)]
    if len(clean) < 2:
        return 0.0
    excess = clean - risk_free_annual / TRADING_DAYS
    sd = np.std(excess, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(excess) / sd * np.sqrt(TRADING_DAYS))


def sortino_ratio(returns: np.ndarray, risk_free_annual: float = 0.0) -> float:
    """Like Sharpe, but only penalizes downside deviation -- upside
    volatility isn't the risk a trader actually minds."""
    clean = returns[np.isfinite(returns)]
    if len(clean) < 2:
        return 0.0
    excess = clean - risk_free_annual / TRADING_DAYS
    downside = excess[excess < 0]
    if len(downside) < 2:
        return 0.0
    downside_sd = np.sqrt(np.mean(downside ** 2))
    if downside_sd == 0:
        return 0.0
    return float(np.mean(excess) / downside_sd * np.sqrt(TRADING_DAYS))


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Largest peak-to-trough decline. Always <= 0."""
    if len(equity_curve) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity_curve)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = equity_curve / peak - 1.0
    finite = dd[np.isfinite(dd)]
    return float(np.min(finite)) if len(finite) else 0.0


def calmar_ratio(equity_curve: np.ndarray, n_days: int) -> float:
    mdd = max_drawdown(equity_curve)
    if mdd == 0:
        return 0.0
    return float(cagr(equity_curve, n_days) / abs(mdd))


def win_rate(trade_pnls: list[float]) -> float:
    closed = [p for p in trade_pnls if p is not None]
    if not closed:
        return 0.0
    return float(sum(1 for p in closed if p > 0) / len(closed))


def profit_factor(trade_pnls: list[float]) -> float:
    closed = [p for p in trade_pnls if p is not None]
    gains = sum(p for p in closed if p > 0)
    losses = -sum(p for p in closed if p < 0)
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def all_metrics(equity_curve: np.ndarray, n_days: int, trade_pnls: list[float]) -> dict:
    returns = daily_returns(equity_curve)
    return {
        "total_return": round(total_return(equity_curve), 4),
        "cagr": round(cagr(equity_curve, n_days), 4),
        "annualized_vol": round(annualized_vol(returns), 4),
        "sharpe": round(sharpe_ratio(returns), 3),
        "sortino": round(sortino_ratio(returns), 3),
        "max_drawdown": round(max_drawdown(equity_curve), 4),
        "calmar": round(calmar_ratio(equity_curve, n_days), 3),
        "win_rate": round(win_rate(trade_pnls), 4),
        "profit_factor": round(profit_factor(trade_pnls), 3) if profit_factor(trade_pnls) != float("inf") else None,
        "n_trades": len([p for p in trade_pnls if p is not None]),
    }
