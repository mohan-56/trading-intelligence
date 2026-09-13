"""Event-driven backtest engine.

The rule that prevents look-ahead structurally, not by convention: a signal
decided using data through bar t fills at bar (t+1)'s OPEN, and the return
attributed to that position is measured open[t+1] -> open[t+2]. The strategy
never sees close[t+1] before its fill is priced -- because the fill uses
open[t+1], which the MarketView it was given (clipped through t) never
contained either.

Costs are charged on turnover, at the point a position changes -- not
"included" as an afterthought. A backtest with silently zero costs is the
single most common way a strategy looks profitable and isn't.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.core.logging import get_logger
from app.domain.backtest.metrics import all_metrics
from app.domain.backtest.models import BacktestResult, BacktestSpec, MarketView, Trade
from app.domain.backtest.risk import RiskEngine
from app.domain.backtest.strategies import Strategy
from app.domain.marketdata.models import BarSeries

log = get_logger("backtest")


@dataclass
class _OpenPosition:
    entry_index: int
    entry_price: float
    weight: float


def run_backtest(
    strategy: Strategy,
    price: BarSeries,
    *,
    auxiliary: dict[str, BarSeries] | None = None,
    risk_engine: RiskEngine | None = None,
    cost_bps: float = 10.0,
    warmup_days: int = 200,
) -> BacktestResult:
    """`price` must be the full history (the engine does its own clipping via
    MarketView -- passing an already-truncated series would just move the
    look-ahead risk to the caller, not remove it). `auxiliary` are extra
    series (e.g. funding) a strategy may ask for by name; each is clipped to
    the same `now` as price, every bar, by this function -- never by the
    strategy itself.
    """
    risk_engine = risk_engine or RiskEngine()
    auxiliary = auxiliary or {}
    n = len(price)
    if n < warmup_days + 3:
        raise ValueError(f"not enough history ({n} bars) for warmup_days={warmup_days}")

    # Seeded to 1.0 by np.ones: equity[warmup_days] and equity[warmup_days+1]
    # are never written by the loop below (the first decision, at
    # t=warmup_days, writes only index warmup_days+2) -- correctly, since no
    # trading has happened yet at those two points.
    #
    # Each iteration of t writes exactly ONE equity index: t+2. This matters
    # more than it looks -- an earlier version wrote BOTH t+1 (a cost-only
    # step) and t+2 (the return step) every iteration, so index t+2 got
    # written once with the real price return by iteration t, then
    # OVERWRITTEN by iteration t+1's cost-only write before that return was
    # ever used. Real BTC history caught it immediately: buy-and-hold from
    # 2018-2026 showed a LOSS with ~0.1% annualized vol, which is impossible
    # for an asset that appreciated roughly 8x with 60-100%+ realized vol
    # every year of that span. Every other equity value was silently
    # collapsing to pure cost drag. One equity index per loop iteration,
    # written once, fixes it.
    equity = np.ones(n)
    weights = np.zeros(n)
    quarantined = 0
    open_position: _OpenPosition | None = None
    trades: list[Trade] = []
    current_weight = 0.0

    for t in range(warmup_days, n - 2):
        now = price.ts[t].astype("datetime64[s]").astype(object)

        if not np.isfinite(price.open[t + 1]) or not np.isfinite(price.open[t + 2]):
            quarantined += 1
            equity[t + 2] = equity[t + 1]  # flat: no trade could execute, no return can accrue
            weights[t + 2] = current_weight
            continue

        view = MarketView(
            now=now,
            series={"price": price.slice_until(now), **{k: v.slice_until(now) for k, v in auxiliary.items()}},
        )
        try:
            signal = strategy.generate_signal(view)
            target_weight = risk_engine.size(signal, view)
        except Exception as exc:  # a broken strategy must not crash the whole run
            log.error("strategy_crashed", strategy=strategy.name, date=str(now), error=str(exc))
            target_weight = 0.0

        fill_price = float(price.open[t + 1])
        turnover = abs(target_weight - current_weight)
        cost = turnover * (cost_bps / 10_000.0)

        # position lifecycle bookkeeping for trade-level stats
        if current_weight == 0.0 and target_weight > 0.0:
            open_position = _OpenPosition(entry_index=t + 1, entry_price=fill_price, weight=target_weight)
        elif current_weight > 0.0 and target_weight == 0.0 and open_position is not None:
            pnl = (fill_price / open_position.entry_price - 1.0) * open_position.weight
            trades.append(Trade(
                entry_date=str(price.ts[open_position.entry_index])[:10], exit_date=str(price.ts[t + 1])[:10],
                entry_price=open_position.entry_price, exit_price=fill_price, weight=open_position.weight, pnl_pct=pnl,
            ))
            open_position = None

        next_return = float(price.open[t + 2] / price.open[t + 1] - 1.0)
        equity[t + 2] = equity[t + 1] * (1.0 - cost) * (1.0 + target_weight * next_return)
        weights[t + 2] = target_weight
        current_weight = target_weight

    if open_position is not None:
        # Still held when the backtest ends -- no exit price exists yet, so
        # pnl_pct is genuinely unknown, not computable-but-forgotten. Report
        # it as an open trade rather than either dropping it (undercounts
        # exposure) or fabricating a close against the last bar's price
        # (that price was never actually traded at).
        trades.append(Trade(
            entry_date=str(price.ts[open_position.entry_index])[:10], exit_date=None,
            entry_price=open_position.entry_price, exit_price=None, weight=open_position.weight, pnl_pct=None,
        ))

    n_days = n - 1 - warmup_days  # elapsed calendar days across equity[warmup_days:]
    metrics = all_metrics(equity[warmup_days:], n_days, [t.pnl_pct for t in trades])

    spec = BacktestSpec(
        strategy_name=strategy.name, symbol=price.symbol.ticker,
        start=str(price.ts[warmup_days])[:10], end=str(price.ts[-1])[:10],
        cost_bps=cost_bps, target_vol_annual=risk_engine.target_vol_annual,
        max_weight=risk_engine.max_weight, warmup_days=warmup_days,
    )
    log.info("backtest_complete", strategy=strategy.name, symbol=price.symbol.ticker,
             n_trades=len(trades), sharpe=metrics["sharpe"], cagr=metrics["cagr"], quarantined=quarantined)
    return BacktestResult(
        spec=spec, dates=[str(d)[:10] for d in price.ts[warmup_days:]],
        equity_curve=equity[warmup_days:].tolist(), weights=weights[warmup_days:].tolist(),
        trades=tuple(trades), metrics=metrics, quarantined_days=quarantined,
    )
