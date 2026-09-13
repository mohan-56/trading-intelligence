"""Injectable clock. Never call datetime.now() directly in domain/ -- time-
dependent logic that can't be frozen can't be tested."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime: ...
    def today(self) -> date: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)

    def today(self) -> date:
        return datetime.now(UTC).date()


class FrozenClock:
    def __init__(self, at: datetime) -> None:
        self._at = at if at.tzinfo else at.replace(tzinfo=UTC)

    def now(self) -> datetime:
        return self._at

    def today(self) -> date:
        return self._at.date()

    def advance(self, **kwargs) -> None:
        self._at += timedelta(**kwargs)


_clock: Clock = SystemClock()


def get_clock() -> Clock:
    return _clock


def set_clock(clock: Clock) -> None:
    global _clock
    _clock = clock


def utcnow() -> datetime:
    return _clock.now()
