"""Ingestion jobs. The ONLY place the system reaches the network. Validate
BEFORE writing -- the lake is trustworthy by construction. Every job is
idempotent: re-running it is always safe.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from app.core.clock import utcnow
from app.core.logging import get_logger
from app.domain.marketdata.quality import QualityReport, check_series
from app.domain.marketdata.service import MarketDataService
from app.domain.symbols.models import Symbol, Timeframe

log = get_logger("ingest")


@dataclass
class IngestResult:
    requested: int = 0
    succeeded: int = 0
    failed: int = 0
    quarantined: int = 0
    rows: int = 0
    reports: list[QualityReport] = field(default_factory=list)
    failures: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "requested": self.requested, "succeeded": self.succeeded, "failed": self.failed,
            "quarantined": self.quarantined, "rows": self.rows, "failures": self.failures,
        }


async def ingest_symbol(
    service: MarketDataService, symbol: Symbol, *, start: datetime, end: datetime,
    timeframe: Timeframe = Timeframe.D1, allow_negative: bool = False,
) -> tuple[bool, QualityReport | None, str]:
    series, _attempts = await service.refresh(symbol, timeframe, start, end, persist=False)
    if not series.is_available:
        return False, None, series.provenance.note or "unavailable"

    report = check_series(series, allow_negative=allow_negative)
    if not report.passed:
        reasons = "; ".join(f"{f.check}({f.count})" for f in report.errors)
        log.error("dq_quarantine", symbol=symbol.ticker, reasons=reasons)
        return False, report, f"quarantined: {reasons}"

    service._lake.write_bars(series)
    if report.warnings:
        log.warning("dq_warnings", symbol=symbol.ticker, warnings=[f.check for f in report.warnings])
    return True, report, ""


async def ingest_universe(
    service: MarketDataService, symbols: list[Symbol], *, years: int = 15,
    timeframe: Timeframe = Timeframe.D1, concurrency: int = 4, allow_negative: bool = False,
) -> IngestResult:
    end = utcnow()
    start = end - timedelta(days=365 * years + 10)
    result = IngestResult(requested=len(symbols))
    semaphore = asyncio.Semaphore(concurrency)

    async def one(symbol: Symbol) -> None:
        async with semaphore:
            try:
                ok, report, reason = await ingest_symbol(
                    service, symbol, start=start, end=end, timeframe=timeframe, allow_negative=allow_negative
                )
            except Exception as exc:
                result.failed += 1
                result.failures[symbol.ticker] = f"{type(exc).__name__}: {exc}"
                log.error("ingest_crashed", symbol=symbol.ticker, error=str(exc))
                return
            if report is not None:
                result.reports.append(report)
            if ok:
                result.succeeded += 1
                result.rows += report.rows if report else 0
            elif report is not None and not report.passed:
                result.quarantined += 1
                result.failures[symbol.ticker] = reason
            else:
                result.failed += 1
                result.failures[symbol.ticker] = reason

    await asyncio.gather(*(one(s) for s in symbols))
    log.info("ingest_complete", succeeded=result.succeeded, failed=result.failed,
             quarantined=result.quarantined, rows=result.rows)
    return result
