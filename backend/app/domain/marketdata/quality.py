"""Data Quality engine. Never silently process corrupted data. Every finding
has a severity: ERROR quarantines the batch, WARNING records and proceeds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from app.domain.marketdata.models import BarSeries


class Severity(StrEnum):
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class Finding:
    check: str
    severity: Severity
    message: str
    count: int = 0
    samples: tuple[str, ...] = ()


@dataclass
class QualityReport:
    symbol: str
    rows: int
    findings: list[Finding] = field(default_factory=list)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.WARNING]

    @property
    def passed(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol, "rows": self.rows, "passed": self.passed,
            "errors": len(self.errors), "warnings": len(self.warnings),
            "findings": [
                {"check": f.check, "severity": f.severity.value, "message": f.message,
                 "count": f.count, "samples": list(f.samples)}
                for f in self.findings
            ],
        }


def _fmt(ts_values: np.ndarray, limit: int = 3) -> tuple[str, ...]:
    return tuple(str(t)[:19] for t in ts_values[:limit])


def check_series(series: BarSeries, *, gap_sigma: float = 12.0, allow_negative: bool = False) -> QualityReport:
    """`allow_negative` is set for derivative/level series where a value below
    zero is real, not corrupt -- e.g. perpetual funding rate is legitimately
    negative when shorts pay longs. A price series never sets this.
    """
    report = QualityReport(symbol=series.symbol.ticker, rows=len(series))
    if series.is_empty:
        report.findings.append(Finding("empty", Severity.ERROR, "series contains no rows"))
        return report

    dup_mask = np.zeros(len(series), dtype=bool)
    dup_mask[1:] = series.ts[1:] == series.ts[:-1]
    if dup_mask.any():
        report.findings.append(Finding("duplicate_timestamps", Severity.ERROR, "duplicate timestamps",
                                       int(dup_mask.sum()), _fmt(series.ts[dup_mask])))

    if len(series) > 1 and (series.ts[1:] < series.ts[:-1]).any():
        report.findings.append(Finding("unsorted", Severity.ERROR, "timestamps not monotonically increasing"))

    non_finite = ~np.isfinite(series.close)
    if non_finite.any():
        report.findings.append(Finding("non_finite_close", Severity.ERROR, "NaN or Inf in close",
                                       int(non_finite.sum()), _fmt(series.ts[non_finite])))

    if not allow_negative:
        non_positive = np.isfinite(series.close) & (series.close <= 0)
        if non_positive.any():
            report.findings.append(Finding("non_positive_price", Severity.ERROR, "close <= 0",
                                           int(non_positive.sum()), _fmt(series.ts[non_positive])))

    finite = np.isfinite(series.open) & np.isfinite(series.high) & np.isfinite(series.low) & np.isfinite(series.close)
    bad_ohlc = finite & (
        (series.high < series.low) | (series.high < series.open) | (series.high < series.close)
        | (series.low > series.open) | (series.low > series.close)
    )
    if bad_ohlc.any():
        report.findings.append(Finding("ohlc_violation", Severity.ERROR, "high/low do not bracket open/close",
                                       int(bad_ohlc.sum()), _fmt(series.ts[bad_ohlc])))

    negative_volume = series.volume < 0
    if negative_volume.any():
        report.findings.append(Finding("negative_volume", Severity.ERROR, "negative volume",
                                       int(negative_volume.sum()), _fmt(series.ts[negative_volume])))

    returns = series.log_returns()[1:]
    clean = returns[np.isfinite(returns)]
    if len(clean) > 30:
        # MAD, not std: a single -40% day inflates its own std enough to hide
        # itself under a std-based threshold. Median absolute deviation is
        # unmoved by the outlier it's trying to detect.
        median = float(np.median(clean))
        mad = float(np.median(np.abs(clean - median)))
        sigma = 1.4826 * mad if mad > 0 else float(np.std(clean))
        if sigma > 0:
            outlier = np.abs(returns) > gap_sigma * sigma
            if outlier.any():
                idx = np.where(outlier)[0] + 1
                report.findings.append(Finding("extreme_move", Severity.WARNING,
                    f"move beyond {gap_sigma} sigma -- possible bad tick or a real event",
                    int(outlier.sum()), _fmt(series.ts[idx])))

        flat = np.abs(returns) < 1e-12
        if len(flat) and flat.mean() > 0.30:
            report.findings.append(Finding("stale_series", Severity.WARNING,
                f"{flat.mean():.0%} of bars have zero change -- series may be stale", int(flat.sum())))

    return report
