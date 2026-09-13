"""Builds provider instances and chains. Small, fixed set for this scope --
no need for a providers.yaml indirection layer with only two real sources.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.core.provenance import Latency
from app.domain.symbols.models import Timeframe
from app.providers.base import MarketDataProvider, ProviderCapabilities
from app.providers.chain import ProviderChain
from app.providers.crypto.binance import (
    BinanceFundingProvider,
    BinanceOpenInterestProvider,
    BinanceProvider,
)
from app.providers.macro.yahoo_context import YahooContextProvider

log = get_logger("providers")


def build_providers() -> dict[str, MarketDataProvider]:
    providers: dict[str, MarketDataProvider] = {
        "binance": BinanceProvider(
            ProviderCapabilities(
                name="binance", kind="crypto", latency=Latency.EOD, trust=95,
                timeframes=(Timeframe.D1, Timeframe.H1, Timeframe.M15),
                min_interval_seconds=0.2, timeout_seconds=15,
            )
        ),
        "binance_funding": BinanceFundingProvider(
            ProviderCapabilities(
                name="binance_funding", kind="crypto_derivatives", latency=Latency.EOD, trust=95,
                min_interval_seconds=0.3, timeout_seconds=15,
            )
        ),
        "binance_oi": BinanceOpenInterestProvider(
            ProviderCapabilities(
                name="binance_oi", kind="crypto_derivatives", latency=Latency.EOD, trust=90,
                min_interval_seconds=0.3, timeout_seconds=15,
            )
        ),
        "yahoo": YahooContextProvider(
            ProviderCapabilities(
                name="yahoo", kind="macro_context", latency=Latency.DELAYED, trust=55,
                min_interval_seconds=1.2, timeout_seconds=25,
            )
        ),
    }
    log.info("providers_built", providers=sorted(providers))
    return providers


def build_chains(providers: dict[str, MarketDataProvider]) -> dict[str, ProviderChain]:
    chains = {
        "crypto": ProviderChain([providers["binance"]]),
        "funding": ProviderChain([providers["binance_funding"]]),
        "open_interest": ProviderChain([providers["binance_oi"]]),
        "context": ProviderChain([providers["yahoo"]]),
    }
    log.info("chains_built", chains={k: v.provider_names for k, v in chains.items()})
    return chains
