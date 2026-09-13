"""Symbol identity. Canonical ticker, exchange-qualified, with a translation
table per provider -- BTCUSD is `BTCUSDT` on Binance, `bitcoin` on CoinGecko.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class AssetClass(StrEnum):
    CRYPTO = "CRYPTO"
    FX = "FX"
    INDEX = "INDEX"  # VIX
    RATE = "RATE"  # US10Y


class Timeframe(StrEnum):
    D1 = "1d"
    H1 = "1h"
    M15 = "15m"
    M5 = "5m"
    M1 = "1m"

    @property
    def seconds(self) -> int:
        return {"1d": 86400, "1h": 3600, "15m": 900, "5m": 300, "1m": 60}[self.value]


@dataclass(frozen=True, slots=True)
class Symbol:
    ticker: str  # canonical, our spelling
    asset_class: AssetClass
    exchange: str
    name: str = ""
    currency: str = "USD"
    provider_tickers: dict[str, str] = field(default_factory=dict, compare=False, hash=False)

    @property
    def key(self) -> str:
        return f"{self.exchange}:{self.ticker}"

    def provider_ticker(self, provider: str) -> str | None:
        return self.provider_tickers.get(provider)

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker, "key": self.key, "name": self.name,
            "asset_class": self.asset_class.value, "exchange": self.exchange, "currency": self.currency,
        }

    def __str__(self) -> str:
        return self.key
