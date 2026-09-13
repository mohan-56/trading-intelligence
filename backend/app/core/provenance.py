"""Provenance -- where a number came from and how much to trust it.

The rule that makes this work: provenance is INSEPARABLE from the value and
combines PESSIMISTICALLY. A figure derived from one live feed and one stale
one inherits the stale one's status -- never the better of the two. Anything
touched by a SIMULATED input stays SIMULATED, transitively, forever.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import IntEnum
from typing import Self


class Latency(IntEnum):
    """Worst-wins ordering: combining provenances takes the MAXIMUM member."""

    LIVE = 0  # streamed, sub-second (WebSocket ticker)
    DELAYED = 1  # known lag
    EOD = 2  # settled end-of-day / settlement price
    HISTORICAL = 3  # backfill, not a current observation
    SIMULATED = 4  # model output, synthetic -- taints everything downstream
    UNAVAILABLE = 5  # explicitly absent. NOT None. NOT 0.0.

    @property
    def label(self) -> str:
        return self.name

    @property
    def is_usable(self) -> bool:
        return self < Latency.UNAVAILABLE


@dataclass(frozen=True, slots=True)
class Provenance:
    sources: tuple[str, ...]
    latency: Latency
    as_of: datetime | None
    fetched_at: datetime | None
    is_cached: bool = False
    note: str = ""

    @classmethod
    def single(
        cls,
        source: str,
        latency: Latency,
        *,
        as_of: datetime | None = None,
        fetched_at: datetime | None = None,
        is_cached: bool = False,
        note: str = "",
    ) -> Self:
        return cls(
            sources=(source,), latency=latency, as_of=as_of,
            fetched_at=fetched_at, is_cached=is_cached, note=note,
        )

    @classmethod
    def unavailable(cls, reason: str, source: str = "none") -> Self:
        return cls(sources=(source,), latency=Latency.UNAVAILABLE, as_of=None, fetched_at=None, note=reason)

    @classmethod
    def combine(cls, provenances: list[Provenance] | tuple[Provenance, ...]) -> Self:
        if not provenances:
            return cls.unavailable("no inputs to combine")
        latency = max(p.latency for p in provenances)
        as_ofs = [p.as_of for p in provenances if p.as_of is not None]
        fetched = [p.fetched_at for p in provenances if p.fetched_at is not None]
        sources = sorted({s for p in provenances for s in p.sources})
        notes = sorted({p.note for p in provenances if p.note})
        return cls(
            sources=tuple(sources), latency=latency,
            as_of=min(as_ofs) if as_ofs else None,
            fetched_at=min(fetched) if fetched else None,
            is_cached=any(p.is_cached for p in provenances),
            note="; ".join(notes),
        )

    def degraded(self, note: str, *, latency: Latency | None = None) -> Self:
        merged = f"{self.note}; {note}" if self.note else note
        return replace(
            self,
            latency=max(self.latency, latency) if latency is not None else self.latency,
            is_cached=True, note=merged,
        )

    def age_seconds(self, now: datetime | None = None) -> float | None:
        if self.as_of is None:
            return None
        return ((now or datetime.now(UTC)) - self.as_of).total_seconds()

    def to_dict(self) -> dict:
        return {
            "sources": list(self.sources),
            "latency": self.latency.label,
            "as_of": self.as_of.isoformat() if self.as_of else None,
            "fetched_at": self.fetched_at.isoformat() if self.fetched_at else None,
            "is_cached": self.is_cached,
            "age_seconds": self.age_seconds(),
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class Tracked[T]:
    value: T | None
    provenance: Provenance = field(default_factory=lambda: Provenance.unavailable("no provenance supplied"))

    @classmethod
    def missing(cls, reason: str, source: str = "none") -> Tracked[T]:
        return cls(value=None, provenance=Provenance.unavailable(reason, source))

    @property
    def is_available(self) -> bool:
        return self.value is not None and self.provenance.latency.is_usable

    @classmethod
    def derive(cls, value: T | None, *inputs: Tracked) -> Tracked[T]:
        prov = Provenance.combine([i.provenance for i in inputs])
        if value is None or not prov.latency.is_usable:
            missing = [i.provenance.note for i in inputs if not i.provenance.latency.is_usable]
            reason = "; ".join(n for n in missing if n) or "derived from unavailable input"
            return cls(value=None, provenance=prov.degraded(reason, latency=Latency.UNAVAILABLE))
        return cls(value=value, provenance=prov)

    def to_dict(self) -> dict:
        return {"value": self.value, "provenance": self.provenance.to_dict()}
