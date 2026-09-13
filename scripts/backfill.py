"""Backfill the crypto data lake: spot OHLCV, funding rate, open interest,
and macro context (DXY, VIX, US10Y).

    python scripts/backfill.py --years 15
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.domain.marketdata.service import MarketDataService
from app.domain.symbols.models import AssetClass
from app.domain.symbols.registry import get_registry
from app.jobs.ingest import ingest_universe
from app.providers.http import close_client
from app.providers.registry import build_chains, build_providers
from app.storage.lake import ParquetLake


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=15)
    args = parser.parse_args()

    configure_logging("INFO")
    settings = get_settings()
    settings.ensure_dirs()

    registry = get_registry()
    providers = build_providers()
    chains = build_chains(providers)

    crypto_symbols = registry.by_asset_class(AssetClass.CRYPTO)
    context_symbols = [s for s in registry.all() if s.asset_class != AssetClass.CRYPTO]

    print(f"\nData root : {settings.data_root}")
    print(f"Crypto    : {[s.ticker for s in crypto_symbols]}")
    print(f"Context   : {[s.ticker for s in context_symbols]}\n")

    t0 = time.perf_counter()

    bars_lake = ParquetLake(series_kind="bars")
    bars_service = MarketDataService(registry, chains["crypto"], bars_lake)
    bars_result = await ingest_universe(bars_service, crypto_symbols, years=args.years)

    context_service = MarketDataService(registry, chains["context"], bars_lake)
    context_result = await ingest_universe(context_service, context_symbols, years=args.years, concurrency=1)

    funding_lake = ParquetLake(series_kind="funding")
    funding_service = MarketDataService(registry, chains["funding"], funding_lake)
    # Binance funding history goes back to each pair's futures listing date,
    # not necessarily 15 years -- request the full window, take what exists.
    funding_result = await ingest_universe(funding_service, crypto_symbols, years=args.years, allow_negative=True)

    oi_lake = ParquetLake(series_kind="open_interest")
    oi_service = MarketDataService(registry, chains["open_interest"], oi_lake)
    oi_result = await ingest_universe(oi_service, crypto_symbols, years=1, concurrency=2)

    elapsed = time.perf_counter() - t0

    print(f"\n{'=' * 74}")
    for name, r in [("bars", bars_result), ("context", context_result),
                     ("funding", funding_result), ("open_interest", oi_result)]:
        print(f"  {name:<14} succeeded {r.succeeded}/{r.requested}   "
              f"failed {r.failed}   quarantined {r.quarantined}   rows {r.rows:,}")
    print(f"\n  elapsed {elapsed:.1f}s")
    bstats = bars_lake.stats()
    print(f"  bars lake: {bstats['files']} files, {bstats['bytes']/1_048_576:.2f} MB")
    print("=" * 74)

    # Kept per-service, not merged -- the same ticker can fail differently in
    # different series (e.g. funding vs open_interest), and merging into one
    # dict by ticker silently drops all but the last reason.
    named_results = [("bars", bars_result), ("context", context_result),
                      ("funding", funding_result), ("open_interest", oi_result)]
    if any(r.failures for _, r in named_results):
        print("\nFAILURES / QUARANTINES")
        for name, r in named_results:
            for ticker, reason in sorted(r.failures.items()):
                print(f"  {name:<14} {ticker:<10} {reason[:90]}")

    warned = [r for res in (bars_result, context_result, funding_result, oi_result) for r in res.reports if r.warnings]
    if warned:
        print(f"\nDATA QUALITY WARNINGS ({len(warned)}) -- recorded, not blocking")
        for r in warned[:15]:
            for f in r.warnings:
                print(f"  {r.symbol:<10} {f.check:<16} {f.message[:55]}  e.g. {', '.join(f.samples)}")

    await close_client()
    return 0 if bars_result.succeeded else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
