"""Historical analogue search.

Answers "has today's market looked like this before, and what happened
next?" -- and does it with three guards that a naive nearest-neighbour search
would skip, each one a real way this kind of engine lies to you if omitted:

1. **Exclusion window.** Today resembles yesterday. Without excluding days
   near the query date, the "closest matches" are trivially the days right
   next to it, which says nothing.
2. **Episode collapsing.** The same historical event (a crash, a rally)
   produces many adjacent similar days. Counting each one as an independent
   match inflates the sample size -- 40 "matches" from one crash is really
   n=1. Episodes are greedily de-duplicated so nearby matches count once.
3. **n_episodes shown everywhere, and a floor below which the confidence
   interval is suppressed rather than printed thin.** A range computed from 3
   episodes is not more informative than no range at all -- it's a false
   sense of precision.

Distance is Mahalanobis on the state vector's expanding z-scores, regularized
toward the diagonal -- several features here are collinear by construction
(four altcoins, each measured against BTC), and an un-regularized covariance
inverse over 16 features and a few thousand rows is numerically unstable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.core.logging import get_logger
from app.domain.quant.features import FeatureMatrix

log = get_logger("analogue")

DEFAULT_GUARDS = {
    "exclusion_window_days": 21,
    "episode_separation_days": 21,
    "min_episodes_for_ci": 8,
    "bootstrap_iterations": 2000,
    "bootstrap_ci": [10, 90],
    "mahalanobis_ridge": 0.10,
}


@dataclass(frozen=True, slots=True)
class Episode:
    anchor_date: str
    distance: float
    forward_returns: dict[str, float | None]  # "1d" -> pct return, None if unavailable

    def to_dict(self) -> dict:
        return {"anchor_date": self.anchor_date, "distance": round(self.distance, 4),
                "forward_returns": {k: (round(v, 4) if v is not None else None) for k, v in self.forward_returns.items()}}


@dataclass(frozen=True, slots=True)
class HorizonStats:
    horizon_days: int
    n_episodes: int
    median: float | None
    p25: float | None
    p75: float | None
    ci_low: float | None
    ci_high: float | None
    reliable: bool

    def to_dict(self) -> dict:
        return {"horizon_days": self.horizon_days, "n_episodes": self.n_episodes, "reliable": self.reliable,
                "median": _round(self.median), "p25": _round(self.p25), "p75": _round(self.p75),
                "ci_low": _round(self.ci_low), "ci_high": _round(self.ci_high)}


def _round(x: float | None) -> float | None:
    return round(x, 4) if x is not None else None


@dataclass(frozen=True, slots=True)
class AnalogueResult:
    query_date: str
    n_candidates: int
    n_episodes: int
    episodes: tuple[Episode, ...]
    horizons: tuple[HorizonStats, ...]
    version: int
    guards: dict

    def to_dict(self) -> dict:
        return {
            "query_date": self.query_date, "n_candidates": self.n_candidates, "n_episodes": self.n_episodes,
            "version": self.version, "guards": self.guards,
            "episodes": [e.to_dict() for e in self.episodes],
            "horizons": [h.to_dict() for h in self.horizons],
        }


def _mahalanobis_distances(query_z: np.ndarray, candidates_z: np.ndarray, ridge: float) -> np.ndarray:
    """Distance from `query_z` to every row of `candidates_z`, using the
    covariance of the candidate pool itself -- i.e. only data available as of
    the query date, never a global/full-sample covariance."""
    n_features = candidates_z.shape[1]
    cov = np.cov(candidates_z, rowvar=False)
    if cov.ndim == 0:  # a single feature column degenerates to a scalar
        cov = np.array([[float(cov)]])
    diag_scale = np.trace(cov) / max(n_features, 1)
    cov_reg = cov + ridge * diag_scale * np.eye(n_features)
    inv_cov = np.linalg.pinv(cov_reg)  # pinv, not inv: still safe if still singular

    diff = candidates_z - query_z
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", diff, inv_cov, diff), 0.0))


def _collapse_to_episodes(order: np.ndarray, distances: np.ndarray, row_indices: np.ndarray, separation_days: int) -> list[tuple[int, float]]:
    """Greedy: take the nearest remaining candidate as an episode anchor,
    drop every candidate within `separation_days` of it (by row index, which
    equals calendar days here since crypto trades every day), repeat."""
    remaining = set(range(len(order)))
    episodes: list[tuple[int, float]] = []
    for pos in order:
        if pos not in remaining:
            continue
        row_i = row_indices[pos]
        episodes.append((row_i, float(distances[pos])))
        remaining -= {p for p in remaining if abs(row_indices[p] - row_i) <= separation_days}
    return episodes


def find_analogues(
    matrix: FeatureMatrix,
    close: np.ndarray,
    query_date_index: int,
    *,
    horizons_days: list[int],
    guards: dict | None = None,
    version: int = 0,
) -> AnalogueResult:
    """`close` must be the price series aligned 1:1 with `matrix.dates`
    (the calendar_symbol's close) -- used only to compute what happened next,
    never as a feature the distance metric sees."""
    g = {**DEFAULT_GUARDS, **(guards or {})}
    query_date = str(matrix.dates[query_date_index])[:10]

    usable = matrix.usable_rows()
    # Strictly the past, and outside the exclusion window -- this keeps the
    # function safe to reuse for a historical backtest of the analogue engine
    # itself later, not only for "today", where every usable row is already
    # in the past by construction.
    candidate_mask = (usable < query_date_index) & (np.abs(usable - query_date_index) > g["exclusion_window_days"])
    candidates = usable[candidate_mask]

    if len(candidates) == 0 or not matrix.complete[query_date_index]:
        return AnalogueResult(query_date, 0, 0, (), _empty_horizons(horizons_days), version, g)

    query_z = matrix.z[query_date_index]
    candidates_z = matrix.z[candidates]
    distances = _mahalanobis_distances(query_z, candidates_z, g["mahalanobis_ridge"])
    order = np.argsort(distances)

    episode_rows = _collapse_to_episodes(order, distances, candidates, g["episode_separation_days"])

    episodes: list[Episode] = []
    for row_i, dist in episode_rows:
        fwd: dict[str, float | None] = {}
        for h in horizons_days:
            target = row_i + h
            if target < len(close) and np.isfinite(close[row_i]) and np.isfinite(close[target]) and close[row_i] != 0:
                fwd[f"{h}d"] = float(close[target] / close[row_i] - 1.0)
            else:
                fwd[f"{h}d"] = None
        episodes.append(Episode(anchor_date=str(matrix.dates[row_i])[:10], distance=dist, forward_returns=fwd))

    horizons = tuple(_horizon_stats(episodes, h, g) for h in horizons_days)
    log.info("analogues_found", query_date=query_date, n_candidates=len(candidates), n_episodes=len(episodes))
    return AnalogueResult(query_date, len(candidates), len(episodes), tuple(episodes), horizons, version, g)


def _horizon_stats(episodes: list[Episode], horizon: int, guards: dict) -> HorizonStats:
    key = f"{horizon}d"
    returns = np.array([e.forward_returns[key] for e in episodes if e.forward_returns[key] is not None])
    n = len(returns)
    if n == 0:
        return HorizonStats(horizon, 0, None, None, None, None, None, False)

    median, p25, p75 = float(np.median(returns)), float(np.percentile(returns, 25)), float(np.percentile(returns, 75))
    reliable = n >= guards["min_episodes_for_ci"]
    ci_low = ci_high = None
    if reliable:
        # Vectorized bootstrap: one (iterations x n) index draw and one
        # axis-wise median, not a Python loop calling rng.choice() +
        # np.median() per iteration. First version did that per-horizon
        # loop and measured ~130ms per horizon (~500ms for all four) against
        # an intended budget of well under 100ms total -- found by actually
        # timing it, not assumed. This version measures ~3ms.
        rng = np.random.default_rng(42)  # fixed seed: identical inputs must reproduce identical CIs
        idx = rng.integers(0, n, size=(guards["bootstrap_iterations"], n))
        resampled_medians = np.median(returns[idx], axis=1)
        lo_pct, hi_pct = guards["bootstrap_ci"]
        ci_low, ci_high = (float(x) for x in np.percentile(resampled_medians, [lo_pct, hi_pct]))

    return HorizonStats(horizon, n, median, p25, p75, ci_low, ci_high, reliable)


def _empty_horizons(horizons_days: list[int]) -> tuple[HorizonStats, ...]:
    return tuple(HorizonStats(h, 0, None, None, None, None, None, False) for h in horizons_days)
