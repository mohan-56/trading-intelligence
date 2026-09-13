"""State vector / feature store.

Builds the daily crypto state vector defined in configs/features.yaml -- the
single input the regime engine (and later, analogue search) consumes.

Three rules this module exists to enforce:

1. **As-of semantics.** Row `t` holds what was KNOWABLE on day `t` -- the last
   known value of every input, never a future one. Funding rate settles 3x a
   day; forward-filling it onto the daily calendar is "what we knew", not an
   invented daily print.

2. **Bounded staleness.** A forward-fill that runs forever turns a dead feed
   into a confident-looking constant. Every series kind has a max staleness;
   past it the feature goes NaN and the row is marked incomplete.

3. **Expanding standardization.** Z-scores use data up to `t` only.
   Full-sample standardization is the most common silent look-ahead bug in
   a system like this, because the output still looks perfectly reasonable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from app.core.config import features_config
from app.core.errors import ConfigError
from app.core.logging import get_logger
from app.core.provenance import Latency, Provenance
from app.domain.marketdata.models import BarSeries
from app.domain.quant import indicators as ind
from app.domain.symbols.models import Timeframe

log = get_logger("features")

# How long a value may be carried forward before we admit we do not know.
MAX_STALENESS_DAYS = {
    "funding": 3,   # settles every 8h; 3 days of silence means the feed is dead
    "bars": 5,      # crypto trades daily; equities-context (Yahoo) can lag a weekend
}


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    name: str
    source: str
    transform: str
    series_kind: str = "bars"
    window: int | None = None
    slow_window: int | None = None
    vs: str | None = None

    @classmethod
    def from_dict(cls, raw: dict) -> FeatureSpec:
        try:
            return cls(
                name=raw["name"], source=raw["source"], transform=raw["transform"],
                series_kind=raw.get("series_kind", "bars"),
                window=raw.get("window"), slow_window=raw.get("slow_window"), vs=raw.get("vs"),
            )
        except KeyError as exc:
            raise ConfigError(f"feature spec missing {exc}: {raw!r}") from exc


@dataclass(frozen=True)
class FeatureMatrix:
    dates: np.ndarray             # datetime64[s], ascending
    names: tuple[str, ...]
    raw: np.ndarray               # (n_dates, n_features), natural units
    z: np.ndarray                 # expanding z-score -- what distance is computed on
    complete: np.ndarray          # bool per row: every feature present
    provenance: Provenance
    version: int

    def __len__(self) -> int:
        return len(self.dates)

    @property
    def n_features(self) -> int:
        return len(self.names)

    def index_of(self, name: str) -> int:
        return self.names.index(name)

    def row_on(self, when: datetime) -> int | None:
        cut = np.datetime64(when.replace(tzinfo=None), "s")
        eligible = np.where(self.dates <= cut)[0]
        return int(eligible[-1]) if len(eligible) else None

    def usable_rows(self) -> np.ndarray:
        return np.where(self.complete & np.isfinite(self.z).all(axis=1))[0]

    def describe_row(self, i: int) -> dict:
        return {
            "date": str(self.dates[i])[:10],
            "complete": bool(self.complete[i]),
            "features": {
                name: {"raw": _f(self.raw[i, j]), "z": _f(self.z[i, j])}
                for j, name in enumerate(self.names)
            },
        }


def _f(x: float) -> float | None:
    return None if not np.isfinite(x) else round(float(x), 6)


def _apply(spec: FeatureSpec, primary: BarSeries, other: BarSeries | None) -> np.ndarray:
    close = primary.close
    w = spec.window
    if spec.transform == "rel_strength" and other is not None:
        # `primary` and `other` are two independent instruments -- SOLUSD has
        # 2,225 rows (listed later on Binance) where BTCUSD has 3,315. Without
        # aligning them first, subtracting one log-return series from the
        # other either raises (mismatched lengths) or, worse, silently lines
        # up index i of one with index i of the other -- comparing SOL's
        # price on day 1 to BTC's price on a *different* calendar day. Same
        # as-of join used for the outer calendar, applied here first.
        other_aligned = _forward_fill_onto(primary.ts, other.ts, other.close, MAX_STALENESS_DAYS["bars"])
        return ind.relative_strength(close, other_aligned, _need(spec, w))
    match spec.transform:
        case "level":
            return close.astype(np.float64)
        case "pct_from_sma":
            return ind.pct_from_sma(close, _need(spec, w))
        case "log_return":
            return ind.log_return(close, _need(spec, w))
        case "diff":
            return ind.diff(close, _need(spec, w))
        case "sma":
            return ind.sma(close, _need(spec, w))
        case "realized_vol":
            return ind.realized_vol(close, _need(spec, w))
        case "vol_ratio":
            return ind.vol_ratio(close, _need(spec, w), spec.slow_window or 90)
        case "expanding_z":
            # A feature can itself be an expanding z-score of the raw series
            # (e.g. funding rate) rather than of the whole assembled matrix --
            # useful when the natural units differ wildly from everything else.
            return ind.expanding_zscore(close, min_periods=30)
        case "rel_strength":
            raise ConfigError(f"{spec.name}: rel_strength needs `vs`")  # handled above when other is present
        case _:
            raise ConfigError(f"{spec.name}: unknown transform '{spec.transform}'")


def _need(spec: FeatureSpec, window: int | None) -> int:
    if window is None:
        raise ConfigError(f"{spec.name}: transform '{spec.transform}' requires `window`")
    return window


def _forward_fill_onto(
    target_dates: np.ndarray, source_dates: np.ndarray, values: np.ndarray, max_staleness_days: int,
) -> np.ndarray:
    """As-of join: for each target date, the last source value at or before
    it, or NaN past `max_staleness_days`. Without that bound, a provider that
    died months ago keeps supplying a confident number today."""
    out = np.full(len(target_dates), np.nan)
    if len(source_dates) == 0:
        return out
    pos = np.searchsorted(source_dates, target_dates, side="right") - 1
    valid = pos >= 0
    if not valid.any():
        return out
    idx = pos[valid]
    candidate = values[idx]
    age_days = (target_dates[valid] - source_dates[idx]).astype("timedelta64[D]").astype(np.int64)
    out[valid] = np.where(age_days <= max_staleness_days, candidate, np.nan)
    return out


class StateVectorBuilder:
    def __init__(self, lakes: dict, config: dict | None = None, registry=None) -> None:
        """`lakes` maps series_kind ('bars' | 'funding' | 'open_interest') to
        an object exposing read_bars(symbol, timeframe) -> BarSeries.

        `registry` resolves ticker -> Symbol and defaults to the real global
        registry; injectable so tests can use fake tickers (SPX, VIX-as-a-
        stub) without needing them to exist in the real crypto universe.
        """
        self._lakes = lakes
        self._config = config or features_config()
        self._registry = registry
        self._specs = [FeatureSpec.from_dict(r) for r in self._config.get("features", [])]
        if not self._specs:
            raise ConfigError("features.yaml defines no features")

    @property
    def specs(self) -> list[FeatureSpec]:
        return list(self._specs)

    @property
    def guards(self) -> dict:
        return self._config.get("guards", {})

    def build(self) -> FeatureMatrix:
        calendar_ticker = self._config.get("calendar_symbol", "BTCUSD")
        calendar = self._read(calendar_ticker, "bars")
        if not calendar.is_available:
            raise ConfigError(f"calendar symbol {calendar_ticker} has no data -- run the backfill first")
        dates = calendar.ts

        cache: dict[tuple[str, str], BarSeries] = {}

        def load(ticker: str, series_kind: str) -> BarSeries:
            key = (ticker, series_kind)
            if key not in cache:
                cache[key] = self._read(ticker, series_kind)
            return cache[key]

        columns: list[np.ndarray] = []
        names: list[str] = []
        provenances: list[Provenance] = []
        missing: list[str] = []

        for spec in self._specs:
            primary = load(spec.source, spec.series_kind)
            other = load(spec.vs, spec.series_kind) if spec.vs else None

            if not primary.is_available or (spec.vs and other is not None and not other.is_available):
                absent = spec.source if not primary.is_available else spec.vs
                missing.append(f"{spec.name} (no {spec.series_kind} data for {absent})")
                columns.append(np.full(len(dates), np.nan))
                names.append(spec.name)
                provenances.append(Provenance.unavailable(f"{spec.name}: {absent} not in lake"))
                continue

            values = _apply(spec, primary, other)
            staleness = MAX_STALENESS_DAYS.get(spec.series_kind, MAX_STALENESS_DAYS["bars"])
            aligned = _forward_fill_onto(dates, primary.ts, values, staleness)

            if not np.isfinite(aligned).any():
                detail = (
                    f"{spec.name}: transform '{spec.transform}' produced no finite values "
                    f"from {spec.source}/{spec.series_kind} ({len(primary)} obs, window={spec.window})."
                )
                log.error("feature_all_nan", feature=spec.name, detail=detail)
                missing.append(detail)

            columns.append(aligned)
            names.append(spec.name)
            provenances.append(
                Provenance.combine([primary.provenance, other.provenance] if other is not None else [primary.provenance])
            )

        raw = np.column_stack(columns) if columns else np.empty((len(dates), 0))
        min_hist = int(self.guards.get("min_history_days", 180))
        z = np.column_stack([ind.expanding_zscore(raw[:, j], min_hist) for j in range(raw.shape[1])]) if raw.shape[1] else raw

        complete = np.isfinite(raw).all(axis=1)
        provenance = Provenance.combine(provenances).degraded(
            f"state vector v{self._config.get('version', 0)}" + (f"; MISSING: {', '.join(missing)}" if missing else ""),
            latency=Latency.EOD,
        )
        if missing:
            log.warning("state_vector_incomplete", missing=missing)
        log.info("state_vector_built", rows=len(dates), features=len(names),
                 usable=int(complete.sum()), version=self._config.get("version"))

        return FeatureMatrix(dates=dates, names=tuple(names), raw=raw, z=z, complete=complete,
                             provenance=provenance, version=int(self._config.get("version", 0)))

    def _read(self, ticker: str, series_kind: str) -> BarSeries:
        lake = self._lakes.get(series_kind)
        if lake is None:
            raise ConfigError(f"no lake configured for series_kind '{series_kind}'")
        registry = self._registry
        if registry is None:
            from app.domain.symbols.registry import get_registry
            registry = get_registry()
        symbol = registry.get(ticker)
        return lake.read_bars(symbol, Timeframe.D1)


def matrix_to_table(matrix: FeatureMatrix):
    import pyarrow as pa

    arrays = [pa.array(matrix.dates.astype("datetime64[s]"), type=pa.timestamp("s"))]
    fields = [pa.field("ts", pa.timestamp("s"))]
    for j, name in enumerate(matrix.names):
        arrays.append(pa.array(matrix.raw[:, j]))
        fields.append(pa.field(name, pa.float64()))
        arrays.append(pa.array(matrix.z[:, j]))
        fields.append(pa.field(f"{name}__z", pa.float64()))
    arrays.append(pa.array(matrix.complete))
    fields.append(pa.field("complete", pa.bool_()))
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))


def save_matrix(matrix: FeatureMatrix, root) -> str:
    import pyarrow.parquet as pq

    out_dir = root / "features" / f"version={matrix.version}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "state_vector.parquet"
    pq.write_table(matrix_to_table(matrix), path, compression="zstd", version="2.6")
    log.info("state_vector_saved", path=str(path), rows=len(matrix))
    return str(path)


def load_matrix(root, version: int) -> FeatureMatrix | None:
    import pyarrow.parquet as pq

    path = root / "features" / f"version={version}" / "state_vector.parquet"
    if not path.exists():
        return None
    table = pq.read_table(path)
    cols = table.column_names
    names = tuple(c for c in cols if not c.endswith("__z") and c not in ("ts", "complete"))
    return FeatureMatrix(
        dates=table.column("ts").to_numpy(zero_copy_only=False).astype("datetime64[s]"),
        names=names,
        raw=np.column_stack([table.column(n).to_numpy(zero_copy_only=False) for n in names]),
        z=np.column_stack([table.column(f"{n}__z").to_numpy(zero_copy_only=False) for n in names]),
        complete=table.column("complete").to_numpy(zero_copy_only=False),
        provenance=Provenance.single("feature_store", Latency.EOD, note=f"state vector v{version} read from lake"),
        version=version,
    )


def latest_snapshot(matrix: FeatureMatrix, when: datetime | None = None) -> dict:
    i = matrix.row_on(when or datetime.now(UTC))
    if i is None:
        return {"available": False, "reason": "no state vector on or before that date"}
    return {"available": True, "version": matrix.version, **matrix.describe_row(i), "provenance": matrix.provenance.to_dict()}
