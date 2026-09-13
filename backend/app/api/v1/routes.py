"""API v1. Thin routers: parse, call domain, serialize. No logic here."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Query, Request

from app.core.errors import DataUnavailable
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
