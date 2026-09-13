"""Regime engine -- rule-based, explainable, evidence-attached.

Deliberately not a black box. Every label comes with the exact feature values
and thresholds that produced it, because a regime call you can't audit is not
meaningfully different from a guess.

Two independent axes, classified separately and then combined:

  TREND        UP / DOWN / SIDEWAYS      -- which family of strategy applies
                                             (trend-following vs mean-reversion)
  VOLATILITY   HIGH / NORMAL / LOW       -- how much room a move needs,
                                             feeds position sizing later

Three more axes are read as SUPPORTING CONTEXT, not folded into the primary
label -- they matter, but conflating five dimensions into one label produces
either an unreadable label or a chosen-by-fiat priority order. Shown as
evidence instead, the way a debate is more honest than a verdict with no
argument attached.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

import numpy as np

from app.domain.quant.features import FeatureMatrix

# Thresholds are on RAW units for interpretability where the unit is already
# meaningful (vol_ratio centres on 1.0 by construction); on Z-SCORE for
# everything else, since z is what's comparable across features and what
# resists needing re-tuning as absolute price levels drift over years.
TREND_Z_THRESHOLD = 0.3
FUNDING_Z_THRESHOLD = 1.0
VOL_RATIO_HIGH = 1.25
VOL_RATIO_LOW = 0.80
ALT_ROTATION_Z_THRESHOLD = 0.3


class Trend(StrEnum):
    UP = "UP"
    DOWN = "DOWN"
    SIDEWAYS = "SIDEWAYS"


class VolState(StrEnum):
    HIGH = "HIGH"
    NORMAL = "NORMAL"
    LOW = "LOW"


class Positioning(StrEnum):
    CROWDED_LONG = "CROWDED_LONG"
    CROWDED_SHORT = "CROWDED_SHORT"
    NEUTRAL = "NEUTRAL"


class AltRotation(StrEnum):
    ALT_SEASON = "ALT_SEASON"       # alts outrunning BTC
    BTC_DOMINANCE = "BTC_DOMINANCE"  # BTC outrunning alts
    NEUTRAL = "NEUTRAL"


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    feature: str
    raw: float | None
    z: float | None
    reads_as: str

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "raw": round(self.raw, 6) if self.raw is not None else None,
            "z": round(self.z, 3) if self.z is not None else None,
            "reads_as": self.reads_as,
        }


@dataclass(frozen=True, slots=True)
class RegimeCall:
    date: str
    trend: Trend
    volatility: VolState
    positioning: Positioning
    alt_rotation: AltRotation
    confidence: float          # 0-1, distance of the driving z-scores from their thresholds
    label: str                 # "UP_HIGH", "SIDEWAYS_NORMAL", ...
    evidence: tuple[EvidenceItem, ...]
    complete: bool             # was every input feature present on this date
    version: int

    def to_dict(self) -> dict:
        return {
            "date": self.date, "trend": self.trend.value, "volatility": self.volatility.value,
            "positioning": self.positioning.value, "alt_rotation": self.alt_rotation.value,
            "confidence": round(self.confidence, 3), "label": self.label,
            "complete": self.complete, "version": self.version,
            "evidence": [e.to_dict() for e in self.evidence],
        }


def _get(matrix: FeatureMatrix, i: int, name: str) -> tuple[float | None, float | None]:
    if name not in matrix.names:
        return None, None
    j = matrix.index_of(name)
    raw, z = matrix.raw[i, j], matrix.z[i, j]
    return (float(raw) if np.isfinite(raw) else None), (float(z) if np.isfinite(z) else None)


def classify_row(matrix: FeatureMatrix, i: int) -> RegimeCall:
    date = str(matrix.dates[i])[:10]
    evidence: list[EvidenceItem] = []

    trend_raw, trend_z = _get(matrix, i, "btc_trend_200d")
    mom_raw, mom_z = _get(matrix, i, "btc_mom_63d")
    if trend_z is not None and mom_z is not None and trend_z > TREND_Z_THRESHOLD and mom_raw is not None and mom_raw > 0:
        trend = Trend.UP
    elif trend_z is not None and mom_z is not None and trend_z < -TREND_Z_THRESHOLD and mom_raw is not None and mom_raw < 0:
        trend = Trend.DOWN
    else:
        trend = Trend.SIDEWAYS
    evidence.append(EvidenceItem("btc_trend_200d", trend_raw, trend_z,
                                 f"{'above' if (trend_raw or 0) > 0 else 'below'} its 200d average"
                                 + (f", z={trend_z:+.2f}" if trend_z is not None else "")))
    evidence.append(EvidenceItem("btc_mom_63d", mom_raw, mom_z,
                                 f"63d return {mom_raw:+.1%}" if mom_raw is not None else "unavailable"))

    vol_raw, vol_z = _get(matrix, i, "btc_vol_ratio")
    if vol_raw is not None and vol_raw > VOL_RATIO_HIGH:
        volatility = VolState.HIGH
    elif vol_raw is not None and vol_raw < VOL_RATIO_LOW:
        volatility = VolState.LOW
    else:
        volatility = VolState.NORMAL
    evidence.append(EvidenceItem("btc_vol_ratio", vol_raw, vol_z,
                                 f"21d/90d realized vol ratio = {vol_raw:.2f}" if vol_raw is not None else "unavailable"))

    fund_raw, fund_z = _get(matrix, i, "btc_funding_z")
    if fund_z is not None and fund_z > FUNDING_Z_THRESHOLD:
        positioning = Positioning.CROWDED_LONG
    elif fund_z is not None and fund_z < -FUNDING_Z_THRESHOLD:
        positioning = Positioning.CROWDED_SHORT
    else:
        positioning = Positioning.NEUTRAL
    evidence.append(EvidenceItem("btc_funding_z", fund_raw, fund_z,
                                 "funding unusually elevated (crowded longs)" if positioning is Positioning.CROWDED_LONG
                                 else "funding unusually negative (crowded shorts)" if positioning is Positioning.CROWDED_SHORT
                                 else "funding within normal range"))

    alt_names = ["eth_vs_btc_63d", "sol_vs_btc_63d", "bnb_vs_btc_63d", "xrp_vs_btc_63d"]
    alt_zs = [z for _, z in (_get(matrix, i, n) for n in alt_names) if z is not None]
    breadth_z = float(np.mean(alt_zs)) if alt_zs else None
    if breadth_z is not None and breadth_z > ALT_ROTATION_Z_THRESHOLD:
        alt_rotation = AltRotation.ALT_SEASON
    elif breadth_z is not None and breadth_z < -ALT_ROTATION_Z_THRESHOLD:
        alt_rotation = AltRotation.BTC_DOMINANCE
    else:
        alt_rotation = AltRotation.NEUTRAL
    evidence.append(EvidenceItem("alt_breadth_avg_z", None, breadth_z,
                                 f"average of {len(alt_zs)}/4 altcoins' relative strength vs BTC"))

    for name in ("dxy_mom_63d", "vix_level", "us10y_chg_63d"):
        raw, z = _get(matrix, i, name)
        evidence.append(EvidenceItem(name, raw, z, "macro context"))

    driving_z = [abs(z) for z in (trend_z, vol_raw and (vol_raw - 1.0) / 0.3) if z is not None]
    confidence = float(min(1.0, np.mean(driving_z) / 1.5)) if driving_z else 0.0

    return RegimeCall(
        date=date, trend=trend, volatility=volatility, positioning=positioning, alt_rotation=alt_rotation,
        confidence=confidence, label=f"{trend.value}_{volatility.value}", evidence=tuple(evidence),
        complete=bool(matrix.complete[i]), version=matrix.version,
    )


def classify_latest(matrix: FeatureMatrix, when: datetime | None = None) -> RegimeCall | None:
    i = matrix.row_on(when or datetime.now(UTC))
    return None if i is None else classify_row(matrix, i)


def classify_history(matrix: FeatureMatrix) -> list[RegimeCall]:
    """Every usable row, classified. Used to build the strategy x regime
    matrix later and to compute how often each label has occurred."""
    return [classify_row(matrix, i) for i in matrix.usable_rows()]
