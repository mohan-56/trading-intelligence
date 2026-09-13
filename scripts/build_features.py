"""Compute the crypto state vector, save it, and classify today's regime.

    python scripts/build_features.py

Run after scripts/backfill.py, and again after any ingest. The API reads the
saved file -- it never computes on the request path.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.domain.quant.features import StateVectorBuilder, latest_snapshot, save_matrix
from app.domain.regime.engine import classify_latest
from app.storage.lake import ParquetLake


def main() -> int:
    configure_logging("INFO")
    get_settings().ensure_dirs()

    lakes = {kind: ParquetLake(series_kind=kind) for kind in ("bars", "funding", "open_interest")}

    t0 = time.perf_counter()
    matrix = StateVectorBuilder(lakes).build()
    elapsed = (time.perf_counter() - t0) * 1000
    save_matrix(matrix, lakes["bars"].root)

    print(f"\n{'=' * 74}")
    print(f"  state vector v{matrix.version}: {len(matrix)} rows x {matrix.n_features} features")
    print(f"  usable rows: {int(matrix.complete.sum())}   built in {elapsed:.0f} ms")
    print(f"  span: {str(matrix.dates[0])[:10]} -> {str(matrix.dates[-1])[:10]}")
    print("=" * 74)

    snap = latest_snapshot(matrix)
    if snap.get("available"):
        print(f"\n  TODAY ({snap['date']})   complete={snap['complete']}\n")
        print(f"  {'feature':<22}{'raw':>16}{'z-score':>10}")
        print(f"  {'-' * 48}")
        for name, vals in snap["features"].items():
            raw = f"{vals['raw']:,.6f}" if vals["raw"] is not None else "-"
            z = f"{vals['z']:+.2f}" if vals["z"] is not None else "-"
            flag = "  <<" if vals["z"] is not None and abs(vals["z"]) >= 1.0 else ""
            print(f"  {name:<22}{raw:>16}{z:>10}{flag}")

        call = classify_latest(matrix)
        if call:
            print(f"\n  REGIME: {call.label}   confidence={call.confidence:.2f}   complete={call.complete}")
            print(f"    trend={call.trend.value}  vol={call.volatility.value}  "
                  f"positioning={call.positioning.value}  alt_rotation={call.alt_rotation.value}")
            print("\n  evidence:")
            for e in call.evidence:
                r = f"{e.raw:,.4f}" if e.raw is not None else "-"
                z = f"z={e.z:+.2f}" if e.z is not None else ""
                print(f"    {e.feature:<20}{r:>14}  {z:<10}  {e.reads_as}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
