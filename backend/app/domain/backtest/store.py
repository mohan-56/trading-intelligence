"""Persist and read backtest results -- the "precompute, then serve" split.

A single backtest is ~700ms (measured; an earlier version was ~1.5s before a
real O(n^2) bug in RiskEngine.size() was fixed -- see risk.py). Four
strategies is ~3s. That is too slow for a request path even after the fix,
so results are computed by scripts/run_backtest.py and read from here --
the API never runs a backtest itself.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.domain.backtest.models import BacktestResult
from app.domain.backtest.strategies import Strategy


def save_results(root: Path, symbol: str, results: list[tuple[Strategy, BacktestResult]]) -> Path:
    out_dir = root / "backtests" / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results.json"
    payload = {
        "symbol": symbol,
        "results": [
            {"strategy": strategy.name, "description": strategy.description, **result.to_dict(max_points=500)}
            for strategy, result in results
        ],
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    tmp.replace(path)
    return path


def load_results(root: Path, symbol: str) -> dict | None:
    path = root / "backtests" / symbol / "results.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
