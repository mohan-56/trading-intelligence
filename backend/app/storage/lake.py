"""Parquet data lake.

Daily/hourly series live in ONE file per symbol per series-kind. Year-
partitioning was tried on a prior build of this system and measured worse:
~250-row-per-year files cost more in per-file overhead than they saved in
partition pruning, for a dataset this size. One file, in-process read cache
invalidated by mtime, covers it.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.provenance import Latency, Provenance
from app.domain.marketdata.models import BarSeries
from app.domain.symbols.models import Symbol, Timeframe

log = get_logger("lake")

BARS_SCHEMA = pa.schema([
    pa.field("ts", pa.timestamp("s")),
    pa.field("open", pa.float64()), pa.field("high", pa.float64()),
    pa.field("low", pa.float64()), pa.field("close", pa.float64()),
    pa.field("volume", pa.float64()),
    pa.field("source", pa.string()), pa.field("latency", pa.string()),
    pa.field("fetched_at", pa.timestamp("s")),
])


class ParquetLake:
    def __init__(self, root: Path | None = None, series_kind: str = "bars") -> None:
        self.root = Path(root) if root else get_settings().lake_dir
        self.series_kind = series_kind
        self.root.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.root / f"_manifest_{series_kind}.json"
        self._cache: dict[tuple[str, str], tuple[float, BarSeries]] = {}
        self._stats_cache: tuple[float, dict] | None = None

    def series_dir(self, symbol: Symbol, timeframe: Timeframe) -> Path:
        return self.root / self.series_kind / f"symbol={symbol.ticker}" / f"tf={timeframe.value}"

    def write_bars(self, series: BarSeries) -> Path | None:
        if series.is_empty:
            return None
        out_dir = self.series_dir(series.symbol, series.timeframe)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "data.parquet"
        table = self._to_table(series)
        if path.exists():
            table = _merge_dedup(pq.read_table(path), table)
        pq.write_table(table, path, compression="zstd", version="2.6")

        self._cache.pop((series.symbol.ticker, series.timeframe.value), None)
        self._stats_cache = None
        self._update_manifest(series, table.num_rows)
        log.info("bars_written", symbol=series.symbol.ticker, timeframe=series.timeframe.value,
                 kind=self.series_kind, rows=table.num_rows)
        return path

    def _to_table(self, series: BarSeries) -> pa.Table:
        n = len(series)
        p = series.provenance
        fetched = p.fetched_at or datetime.now(UTC)
        return pa.Table.from_arrays([
            pa.array(series.ts.astype("datetime64[s]"), type=pa.timestamp("s")),
            pa.array(series.open), pa.array(series.high), pa.array(series.low),
            pa.array(series.close), pa.array(series.volume),
            pa.array([",".join(p.sources)] * n, type=pa.string()),
            pa.array([p.latency.label] * n, type=pa.string()),
            pa.array([np.datetime64(fetched.replace(tzinfo=None), "s")] * n, type=pa.timestamp("s")),
        ], schema=BARS_SCHEMA)

    def read_bars(
        self, symbol: Symbol, timeframe: Timeframe = Timeframe.D1,
        start: datetime | None = None, end: datetime | None = None,
    ) -> BarSeries:
        out_dir = self.series_dir(symbol, timeframe)
        path = out_dir / "data.parquet"
        if not path.exists():
            return BarSeries.unavailable(symbol, timeframe, f"no stored {self.series_kind} for {symbol.ticker}")

        cache_key = (symbol.ticker, timeframe.value)
        mtime = path.stat().st_mtime
        hit = self._cache.get(cache_key)
        if hit is not None and hit[0] == mtime:
            full = hit[1]
            return full.slice_range(start, end) if (start or end) else full

        table = pq.read_table(path).sort_by("ts")
        if table.num_rows == 0:
            return BarSeries.unavailable(symbol, timeframe, "stored file was empty")

        sources = tuple(sorted({s for s in table.column("source").to_pylist() if s}))
        latencies = {str(x) for x in table.column("latency").to_pylist()}
        worst = max((Latency[x] for x in latencies if x in Latency.__members__), default=Latency.HISTORICAL)
        fetched = table.column("fetched_at").to_pylist()
        ts = table.column("ts").to_numpy(zero_copy_only=False).astype("datetime64[s]")

        series = BarSeries(
            symbol=symbol, timeframe=timeframe, ts=ts,
            open=table.column("open").to_numpy(zero_copy_only=False),
            high=table.column("high").to_numpy(zero_copy_only=False),
            low=table.column("low").to_numpy(zero_copy_only=False),
            close=table.column("close").to_numpy(zero_copy_only=False),
            volume=table.column("volume").to_numpy(zero_copy_only=False),
            provenance=Provenance(
                sources=sources or ("lake",), latency=worst,
                as_of=_utc(ts[-1].astype(datetime)),
                fetched_at=_utc(min(f for f in fetched if f)) if any(fetched) else None,
                is_cached=True, note=f"read from local lake ({self.series_kind})",
            ),
        )
        self._cache[cache_key] = (mtime, series)
        return series.slice_range(start, end) if (start or end) else series

    def has_bars(self, symbol: Symbol, timeframe: Timeframe = Timeframe.D1) -> bool:
        return (self.series_dir(symbol, timeframe) / "data.parquet").exists()

    def stats(self, ttl_seconds: float = 60.0) -> dict:
        now = time.monotonic()
        if self._stats_cache is not None and now - self._stats_cache[0] < ttl_seconds:
            return self._stats_cache[1]
        files = list(self.root.rglob("*.parquet"))
        value = {"root": str(self.root), "files": len(files), "bytes": sum(f.stat().st_size for f in files)}
        self._stats_cache = (now, value)
        return value

    def _update_manifest(self, series: BarSeries, rows: int) -> None:
        manifest = self.manifest()
        manifest[series.symbol.ticker] = {
            "ticker": series.symbol.ticker, "timeframe": series.timeframe.value, "rows": rows,
            "start": series.start.isoformat() if series.start else None,
            "end": series.end.isoformat() if series.end else None,
            "provenance": series.provenance.to_dict(),
        }
        tmp = self._manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        tmp.replace(self._manifest_path)

    def manifest(self) -> dict:
        if not self._manifest_path.exists():
            return {}
        try:
            return json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}


def _merge_dedup(existing: pa.Table, incoming: pa.Table) -> pa.Table:
    combined = pa.concat_tables([existing.cast(BARS_SCHEMA), incoming.cast(BARS_SCHEMA)])
    order = pa.compute.sort_indices(combined, sort_keys=[("ts", "ascending"), ("fetched_at", "descending")])
    combined = combined.take(order)
    ts = combined.column("ts").to_pylist()
    keep = [i for i in range(len(ts)) if i == 0 or ts[i] != ts[i - 1]]
    return combined.take(pa.array(keep))


def _utc(value) -> datetime:
    dt = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt
