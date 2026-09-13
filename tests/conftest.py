"""Test fixtures. The whole suite runs OFFLINE and deterministically."""

from __future__ import annotations

import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

_TMP_ROOT = Path(tempfile.mkdtemp(prefix="ci-test-"))
os.environ["TI_DATA_ROOT"] = str(_TMP_ROOT).replace("\\", "/")
os.environ["TI_ENV"] = "test"
os.environ["TI_LOG_LEVEL"] = "WARNING"

from app.core.clock import FrozenClock, set_clock  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.domain.symbols.models import AssetClass, Symbol  # noqa: E402
from app.domain.symbols.registry import SymbolRegistry, get_registry  # noqa: E402
from app.storage.lake import ParquetLake  # noqa: E402

FROZEN_NOW = datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def frozen_clock():
    clock = FrozenClock(FROZEN_NOW)
    set_clock(clock)
    yield clock


@pytest.fixture
def settings():
    return get_settings()


@pytest.fixture
def tmp_lake(tmp_path: Path) -> ParquetLake:
    return ParquetLake(root=tmp_path / "lake", series_kind="bars")


@pytest.fixture
def registry() -> SymbolRegistry:
    return get_registry()


@pytest.fixture
def btc_symbol() -> Symbol:
    return Symbol(
        ticker="BTCUSD", asset_class=AssetClass.CRYPTO, exchange="BINANCE", name="Bitcoin",
        provider_tickers={"binance": "BTCUSDT", "binance_futures": "BTCUSDT"},
    )
