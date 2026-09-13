"""API v1. Thin routers: parse, call domain, serialize. No logic here."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
from fastapi import APIRouter, Query, Request

from app.core.config import features_config
from app.core.errors import DataUnavailable
from app.domain.analogue.engine import find_analogues
from app.domain.backtest.store import load_results
from app.domain.quant.features import latest_snapshot
from app.domain.regime.engine import classify_history, classify_latest
from app.domain.symbols.models import Timeframe

router = APIRouter(prefix="/api/v1")


@router.get("/health", tags=["system"])
async def health() -> dict:
    return {"status": "ok", "time": datetime.now(UTC).isoformat()}


@router.get("/status", tags=["system"])
async def status(request: Request) -> dict:
    chains = request.app.state.chains
    return {
        "time": datetime.now(UTC).isoformat(),
        "chains": {name: {"providers": c.provider_names, "breakers": c.health()} for name, c in chains.items()},
        "lakes": {kind: lake.stats() for kind, lake in request.app.state.lakes.items()},
    }


@router.get("/symbols", tags=["reference"])
async def list_symbols(request: Request) -> dict:
    symbols = request.app.state.registry.all()
    return {"count": len(symbols), "symbols": [s.to_dict() for s in symbols]}


@router.get("/coverage", tags=["reference"])
async def coverage(request: Request) -> dict:
    rows = request.app.state.marketdata["bars"].coverage()
    return {"count": len(rows), "with_data": sum(1 for r in rows if r["rows"] > 0), "symbols": rows}


@router.get("/bars/{ticker}", tags=["market-data"])
async def get_bars(
    request: Request, ticker: str, kind: str = "bars",
    timeframe: Timeframe = Timeframe.D1, start: datetime | None = None, end: datetime | None = None,
    max_points: int = Query(1000, ge=10, le=20000),
) -> dict:
    service = request.app.state.marketdata.get(kind)
    if service is None:
        raise DataUnavailable(f"unknown series kind '{kind}'", detail={"available": list(request.app.state.marketdata)})
    series = service.get_bars(ticker, timeframe, start, end)
    if not series.is_available:
        raise DataUnavailable(f"no {kind} data for {ticker}", detail={"provenance": series.provenance.to_dict()})
    return {**series.describe(), "bars": series.to_records(max_points=max_points)}


@router.get("/regime/latest", tags=["intelligence"])
async def regime_latest(request: Request, when: datetime | None = None) -> dict:
    """Today's regime call: trend x volatility, plus positioning, alt
    rotation and macro context as supporting evidence. Read from the saved
    state vector -- never computed on the request path."""
    matrix = request.app.state.features
    if matrix is None:
        raise DataUnavailable("state vector not built", detail={"fix": "python scripts/build_features.py"})
    call = classify_latest(matrix, when)
    if call is None:
        raise DataUnavailable("no regime call for that date")
    return call.to_dict()


@router.get("/regime/history", tags=["intelligence"])
async def regime_history(request: Request, max_points: int = Query(500, ge=10, le=5000)) -> dict:
    """Every usable historical regime call -- what the engine would have
    said on each past day, using only data available as of that day."""
    matrix = request.app.state.features
    if matrix is None:
        raise DataUnavailable("state vector not built", detail={"fix": "python scripts/build_features.py"})
    calls = classify_history(matrix)
    if len(calls) > max_points:
        idx = np.unique(np.linspace(0, len(calls) - 1, max_points).astype(int))
        calls = [calls[i] for i in idx]
    return {"count": len(calls), "version": matrix.version, "calls": [c.to_dict() for c in calls]}


@router.get("/state-vector/latest", tags=["intelligence"])
async def state_vector_latest(request: Request, when: datetime | None = None) -> dict:
    matrix = request.app.state.features
    if matrix is None:
        raise DataUnavailable("state vector not built", detail={"fix": "python scripts/build_features.py"})
    snapshot = latest_snapshot(matrix, when)
    if not snapshot.get("available"):
        raise DataUnavailable(snapshot.get("reason", "no state vector for that date"))
    return snapshot


@router.get("/analogues/latest", tags=["intelligence"])
async def analogues_latest(request: Request, when: datetime | None = None) -> dict:
    """Historical episodes that resembled the query date, with what
    happened next. Computed on request -- measured over real HTTP against
    ~1,960 candidates: ~51ms of actual computation, ~107ms median /
    ~184ms p95 end to end once lake reads and JSON serialization are
    included. First version bootstrapped with a per-iteration Python loop
    and measured 582ms; vectorizing it got computation to ~51ms (see
    analogue/engine.py). If the remaining request overhead becomes a
    problem as the universe grows, this moves into the nightly precompute
    like regime -- not yet necessary at this size."""
    matrix = request.app.state.features
    if matrix is None:
        raise DataUnavailable("state vector not built", detail={"fix": "python scripts/build_features.py"})

    i = matrix.row_on(when or datetime.now(UTC))
    if i is None:
        raise DataUnavailable("no state vector for that date")

    calendar_ticker = features_config().get("calendar_symbol", "BTCUSD")
    bars_service = request.app.state.marketdata.get("bars")
    close_series = bars_service.get_bars(calendar_ticker)
    if not close_series.is_available:
        raise DataUnavailable(f"no price history for calendar symbol {calendar_ticker}")

    # `close_series` is read live from the lake; `matrix` was loaded from a
    # saved snapshot at startup. If bars were ingested since the state vector
    # was last built, the two can silently drift out of alignment -- and
    # find_analogues assumes index i means the same calendar day in both.
    # Fail loudly rather than compute a forward return against the wrong day.
    if len(close_series) < len(matrix) or not np.array_equal(close_series.ts[: len(matrix)], matrix.dates):
        raise DataUnavailable(
            "state vector is out of sync with the price lake",
            detail={"fix": "python scripts/build_features.py", "lake_rows": len(close_series), "matrix_rows": len(matrix)},
        )

    result = find_analogues(
        matrix, close_series.close, i,
        horizons_days=features_config().get("forward_horizons_days", [1, 3, 7, 21]),
        guards=features_config().get("guards", {}),
        version=matrix.version,
    )
    return result.to_dict()


@router.get("/backtest/strategies", tags=["backtest"])
async def backtest_strategies(request: Request, symbol: str = "BTCUSD") -> dict:
    """Reads precomputed results -- never runs a backtest on a request.

    First version computed all 4 strategies live: measured 27 seconds end to
    end. Profiled it: 79% of a single backtest's time was RiskEngine.size()
    recomputing a full rolling-volatility series over the ENTIRE (ever-
    growing) clipped history at every bar, just to read its last value --
    O(n) of wasted work per bar, O(n^2) over the run. Slicing to the small
    tail the calculation actually needs (risk.py) cut a single backtest from
    ~1.5s to ~0.7s -- real, but still ~3s for four strategies, still too
    slow for a request. Moved to the same precompute-then-serve pattern as
    regime and the state vector: scripts/run_backtest.py runs it, saves it,
    and this endpoint only ever reads the file.
    """
    bars_lake = request.app.state.lakes.get("bars")
    saved = load_results(bars_lake.root, symbol)
    if saved is None:
        raise DataUnavailable(
            f"no precomputed backtest for {symbol}",
            detail={"fix": f"python scripts/run_backtest.py --symbol {symbol}"},
        )
    return saved
