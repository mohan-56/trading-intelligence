"""Symbol registry -- loads configs/universe.yaml, resolves provider spellings."""

from __future__ import annotations

import functools

from app.core.config import universe_config
from app.core.errors import ConfigError, SymbolNotFound
from app.domain.symbols.models import AssetClass, Symbol


class SymbolRegistry:
    def __init__(self, symbols: list[Symbol]) -> None:
        self._by_ticker: dict[str, Symbol] = {}
        self._by_key: dict[str, Symbol] = {}
        for s in symbols:
            if s.ticker in self._by_ticker:
                raise ConfigError(f"duplicate canonical ticker in universe: {s.ticker}")
            self._by_ticker[s.ticker] = s
            self._by_key[s.key] = s

    @classmethod
    def from_config(cls) -> SymbolRegistry:
        cfg = universe_config()
        entries = cfg.get("symbols") or []
        if not entries:
            raise ConfigError("universe.yaml contains no symbols")
        symbols: list[Symbol] = []
        for raw in entries:
            try:
                symbols.append(
                    Symbol(
                        ticker=raw["ticker"],
                        asset_class=AssetClass(raw["asset_class"]),
                        exchange=raw["exchange"],
                        name=raw.get("name", ""),
                        currency=raw.get("currency", "USD"),
                        provider_tickers=dict(raw.get("providers") or {}),
                    )
                )
            except (KeyError, ValueError) as exc:
                raise ConfigError(f"bad universe entry {raw!r}: {exc}") from exc
        return cls(symbols)

    def get(self, ticker: str) -> Symbol:
        sym = self._by_ticker.get(ticker) or self._by_key.get(ticker)
        if sym is None:
            raise SymbolNotFound(f"unknown symbol: {ticker}")
        return sym

    def all(self) -> list[Symbol]:
        return list(self._by_ticker.values())

    def by_asset_class(self, asset_class: AssetClass) -> list[Symbol]:
        return [s for s in self._by_ticker.values() if s.asset_class == asset_class]

    def supported_by(self, provider: str) -> list[Symbol]:
        return [s for s in self._by_ticker.values() if provider in s.provider_tickers]

    def __len__(self) -> int:
        return len(self._by_ticker)

    def __contains__(self, ticker: object) -> bool:
        return isinstance(ticker, str) and (ticker in self._by_ticker or ticker in self._by_key)


@functools.lru_cache(maxsize=1)
def get_registry() -> SymbolRegistry:
    return SymbolRegistry.from_config()
