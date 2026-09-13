"""Regression guard for the routing bug the first real backfill found:
BinanceFundingProvider and BinanceOpenInterestProvider look up a symbol's
'binance_futures' mapping internally, but the base class's default
`supports_symbol` checked `provider_ticker(self.name)` -- "binance_funding" /
"binance_oi" -- which no symbol in the universe actually has. Every funding
and open-interest fetch failed with "symbol not mapped" before get_bars ever
ran, and it looked like a real API problem until traced.
"""

from __future__ import annotations

from app.core.provenance import Latency
from app.domain.symbols.models import AssetClass, Symbol
from app.providers.base import ProviderCapabilities
from app.providers.crypto.binance import (
    BinanceFundingProvider,
    BinanceOpenInterestProvider,
    BinanceProvider,
)


def _caps(name: str) -> ProviderCapabilities:
    return ProviderCapabilities(name=name, kind="crypto", latency=Latency.EOD, min_interval_seconds=0.0)


class TestSupportsSymbolRouting:
    def test_funding_provider_uses_the_futures_mapping(self, btc_symbol: Symbol):
        provider = BinanceFundingProvider(_caps("binance_funding"))
        assert provider.supports_symbol(btc_symbol), (
            "supports_symbol must check 'binance_futures', not self.name -- "
            "this is exactly the bug the first real backfill hit"
        )

    def test_oi_provider_uses_the_futures_mapping(self, btc_symbol: Symbol):
        provider = BinanceOpenInterestProvider(_caps("binance_oi"))
        assert provider.supports_symbol(btc_symbol)

    def test_symbol_without_futures_mapping_is_unsupported(self):
        spot_only = Symbol(
            ticker="OBSCURE", asset_class=AssetClass.CRYPTO, exchange="BINANCE",
            provider_tickers={"binance": "OBSCUREUSDT"},  # no binance_futures key
        )
        assert not BinanceFundingProvider(_caps("binance_funding")).supports_symbol(spot_only)
        assert not BinanceOpenInterestProvider(_caps("binance_oi")).supports_symbol(spot_only)

    def test_spot_provider_still_uses_default_self_name_check(self, btc_symbol: Symbol):
        """Spot klines aren't affected -- self.name is 'binance', which does
        match a real key in the universe. This test pins that the fix didn't
        accidentally change the (correct) default behaviour for spot."""
        assert BinanceProvider(_caps("binance")).supports_symbol(btc_symbol)
