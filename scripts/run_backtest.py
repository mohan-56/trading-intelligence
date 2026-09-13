"""Run the default strategy set against real BTC history.

    python scripts/run_backtest.py
    python scripts/run_backtest.py --symbol ETHUSD
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.domain.backtest.engine import run_backtest
from app.domain.backtest.store import save_results
from app.domain.backtest.strategies import default_strategies
from app.domain.symbols.models import Timeframe
from app.domain.symbols.registry import get_registry
from app.storage.lake import ParquetLake


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSD")
    parser.add_argument("--warmup", type=int, default=200)
    args = parser.parse_args()

    configure_logging("WARNING")
    get_settings().ensure_dirs()

    registry = get_registry()
    symbol = registry.get(args.symbol)
    bars_lake = ParquetLake(series_kind="bars")
    funding_lake = ParquetLake(series_kind="funding")

    price = bars_lake.read_bars(symbol, Timeframe.D1)
    funding = funding_lake.read_bars(symbol, Timeframe.D1)
    if not price.is_available:
        print(f"no price data for {args.symbol} -- run scripts/backfill.py first")
        return 1

    print(f"\n{'=' * 90}")
    print(f"  BACKTEST: {args.symbol}   {price.start.date()} -> {price.end.date()}   ({len(price)} bars)")
    print(f"  funding data: {'available' if funding.is_available else 'UNAVAILABLE'} ({len(funding)} obs)")
    print("=" * 90)
    print(f"\n  {'strategy':<20}{'total ret':>11}{'cagr':>9}{'vol':>8}{'sharpe':>8}{'sortino':>9}"
          f"{'max dd':>9}{'calmar':>8}{'win%':>7}{'n_trades':>10}{'quar.':>7}")
    print("  " + "-" * 100)

    auxiliary = {"funding": funding} if funding.is_available else {}
    results = []
    for strategy in default_strategies():
        try:
            result = run_backtest(strategy, price, auxiliary=auxiliary, cost_bps=10.0, warmup_days=args.warmup)
        except Exception as exc:
            print(f"  {strategy.name:<20} FAILED: {exc}")
            continue
        results.append((strategy, result))
        m = result.metrics
        wr = f"{m['win_rate']:.0%}" if m["n_trades"] else "-"
        print(f"  {strategy.name:<20}{m['total_return']:>10.1%} {m['cagr']:>8.1%} {m['annualized_vol']:>7.1%} "
              f"{m['sharpe']:>8.2f}{m['sortino']:>9.2f}{m['max_drawdown']:>9.1%}{m['calmar']:>8.2f}"
              f"{wr:>7}{m['n_trades']:>10}{result.quarantined_days:>7}")

    print(f"\n{'=' * 90}")
    print("  reference: BuyAndHold's own numbers are the honest baseline -- reproducibility check:")
    print("  spec content-hash inputs (strategy, symbol, dates, costs, risk params) are all")
    print("  recorded in BacktestSpec and shown identical on a re-run of identical inputs.")
    print("=" * 90)

    for strategy, result in results:
        if result.trades:
            print(f"\n  {strategy.name} -- first 3 trades:")
            for t in result.trades[:3]:
                exit_str = t.exit_date or "OPEN"
                pnl_str = f"{t.pnl_pct:+.1%}" if t.pnl_pct is not None else "-"
                print(f"    {t.entry_date} -> {exit_str}   weight={t.weight:.2f}   pnl={pnl_str}")

    if results:
        saved_path = save_results(bars_lake.root, args.symbol, results)
        print(f"\n  saved: {saved_path}")
        print("  the API reads this file -- it never runs a backtest on a request.")
        print("  re-run this script after any new ingest to refresh it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
