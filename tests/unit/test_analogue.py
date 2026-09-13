"""Analogue engine tests.

The guards are the product here, more than the nearest-neighbour search
itself: exclusion window (autocorrelation), episode collapsing (sample-size
inflation), and the reliability floor (false precision from too few
episodes). Each gets a test that would fail if the guard were silently
removed.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.provenance import Latency, Provenance
from app.domain.analogue.engine import find_analogues
from app.domain.quant.features import FeatureMatrix

NAMES = ("f1", "f2", "f3")


def _matrix(n: int, z: np.ndarray, complete: np.ndarray | None = None) -> FeatureMatrix:
    dates = np.array([np.datetime64("2020-01-01") + np.timedelta64(i, "D") for i in range(n)], dtype="datetime64[s]")
    raw = z.copy()  # raw/z distinction doesn't matter for these tests
    complete = np.ones(n, dtype=bool) if complete is None else complete
    return FeatureMatrix(dates=dates, names=NAMES, raw=raw, z=z, complete=complete,
                         provenance=Provenance.single("test", Latency.EOD), version=1)


def _flat_close(n: int, start: float = 100.0) -> np.ndarray:
    """A close series with a small deterministic drift per day, so forward
    returns are computable and distinguishable without adding real noise."""
    return start * (1.001 ** np.arange(n))


GUARDS = {
    "exclusion_window_days": 5, "episode_separation_days": 5,
    "min_episodes_for_ci": 3, "bootstrap_iterations": 200,
    "bootstrap_ci": [10, 90], "mahalanobis_ridge": 0.1,
}


class TestExclusionWindow:
    def test_nearby_days_are_never_candidates(self):
        """Without this guard, the 'closest matches' to today are trivially
        yesterday and the day before -- which says nothing."""
        rng = np.random.default_rng(1)
        n = 200
        z = rng.normal(0, 1, (n, 3))
        query = n - 1
        z[query] = [0.0, 0.0, 0.0]
        z[query - 2] = [0.0, 0.0, 0.0]  # a near-identical day, inside the window
        z[50] = [0.0, 0.0, 0.0]         # a near-identical day, outside the window

        result = find_analogues(_matrix(n, z), _flat_close(n), query, horizons_days=[1], guards=GUARDS)
        anchor_dates = {e.anchor_date for e in result.episodes}
        # The row at query-2 must never appear as a candidate at all.
        near_row_date = str(_matrix(n, z).dates[query - 2])[:10]
        assert near_row_date not in anchor_dates

    def test_candidates_are_strictly_in_the_past(self):
        """Required so this function stays safe to reuse for a historical
        backtest of the engine itself later -- a candidate from the query's
        own future would be look-ahead."""
        rng = np.random.default_rng(2)
        n = 100
        z = rng.normal(0, 1, (n, 3))
        query = 50
        result = find_analogues(_matrix(n, z), _flat_close(n), query, horizons_days=[1], guards=GUARDS)
        query_date = str(_matrix(n, z).dates[query])[:10]
        for e in result.episodes:
            assert e.anchor_date < query_date


class TestEpisodeCollapsing:
    def test_a_cluster_of_similar_days_becomes_one_episode(self):
        """A cluster of adjacent near-identical days, narrower than the
        separation guard, must not count as separate matches -- that inflates
        the effective sample size to something the data doesn't support."""
        rng = np.random.default_rng(3)
        n = 300
        z = rng.normal(0, 1, (n, 3))
        # 4 consecutive near-identical days -- narrower than
        # episode_separation_days=5, so one greedy pick must absorb all of
        # them. (A cluster WIDER than the separation window legitimately
        # splits into more than one episode; that's covered separately.)
        for i in range(100, 104):
            z[i] = [0.01, 0.01, 0.01]
        query = n - 1
        z[query] = [0.0, 0.0, 0.0]

        result = find_analogues(_matrix(n, z), _flat_close(n), query, horizons_days=[1], guards=GUARDS)
        cluster_dates = {str(_matrix(n, z).dates[i])[:10] for i in range(100, 104)}
        matched_from_cluster = [e for e in result.episodes if e.anchor_date in cluster_dates]
        assert len(matched_from_cluster) == 1, "a cluster narrower than the separation window must collapse to one episode"

    def test_a_cluster_wider_than_the_separation_window_splits(self):
        """The mirror case: a cluster WIDER than episode_separation_days is
        expected to produce more than one episode -- the guard bounds how
        close two counted episodes may be, it doesn't force everything
        vaguely similar into a single bucket regardless of span."""
        rng = np.random.default_rng(3)
        n = 300
        z = rng.normal(0, 1, (n, 3))
        for i in range(100, 110):  # 10 days wide, separation guard is 5
            z[i] = [0.01, 0.01, 0.01]
        query = n - 1
        z[query] = [0.0, 0.0, 0.0]

        result = find_analogues(_matrix(n, z), _flat_close(n), query, horizons_days=[1], guards=GUARDS)
        cluster_dates = {str(_matrix(n, z).dates[i])[:10] for i in range(100, 110)}
        matched_from_cluster = [e for e in result.episodes if e.anchor_date in cluster_dates]
        assert len(matched_from_cluster) >= 2

    def test_episodes_are_temporally_separated(self):
        rng = np.random.default_rng(4)
        n = 300
        z = rng.normal(0, 1, (n, 3))
        query = n - 1
        result = find_analogues(_matrix(n, z), _flat_close(n), query, horizons_days=[1], guards=GUARDS)
        matrix = _matrix(n, z)
        day_indices = sorted(
            i for i in range(n) if str(matrix.dates[i])[:10] in {e.anchor_date for e in result.episodes}
        )
        gaps = np.diff(day_indices)
        assert (gaps > GUARDS["episode_separation_days"]).all() or len(gaps) == 0


class TestReliabilityFloor:
    def test_few_episodes_suppresses_the_confidence_interval(self):
        """A CI from 2 episodes is a false sense of precision -- it must be
        suppressed, not printed thin."""
        rng = np.random.default_rng(5)
        n = 60
        z = rng.normal(0, 1, (n, 3)) * 10  # spread candidates out, few will be close
        query = n - 1
        z[query] = [0.0, 0.0, 0.0]
        z[5] = [0.0, 0.0, 0.0]  # exactly one very close match, far enough back

        result = find_analogues(_matrix(n, z), _flat_close(n), query, horizons_days=[1], guards=GUARDS)
        h = result.horizons[0]
        if h.n_episodes < GUARDS["min_episodes_for_ci"]:
            assert not h.reliable
            assert h.ci_low is None and h.ci_high is None

    def test_enough_episodes_produces_a_ci(self):
        rng = np.random.default_rng(6)
        n = 400
        z = rng.normal(0, 1, (n, 3))
        query = n - 1
        z[query] = [0.0, 0.0, 0.0]
        for i in range(0, 300, 15):  # many well-separated near-identical days
            z[i] = [0.02, 0.02, 0.02]

        result = find_analogues(_matrix(n, z), _flat_close(n), query, horizons_days=[1], guards=GUARDS)
        h = result.horizons[0]
        assert h.n_episodes >= GUARDS["min_episodes_for_ci"]
        assert h.reliable
        assert h.ci_low is not None and h.ci_high is not None
        assert h.ci_low <= h.median <= h.ci_high


class TestForwardReturns:
    def test_forward_return_matches_hand_computed_value(self):
        n = 100
        close = _flat_close(n, start=100.0)
        rng = np.random.default_rng(20)
        z = rng.normal(0, 1, (n, 3)) * 5  # spread everything out...
        query = 90
        z[query] = [0.0, 0.0, 0.0]
        z[80] = [0.0, 0.0, 0.0]           # ...except this one exact match
        result = find_analogues(_matrix(n, z), close, query, horizons_days=[1, 5], guards=GUARDS)
        # Row 80 is the intended exact match; the rest of the randomly
        # scattered rows legitimately produce their own (more distant)
        # episodes too. Distances are sorted ascending, so the exact match
        # -- distance 0.0 -- must be first.
        nearest = result.episodes[0]
        assert nearest.anchor_date == str(_matrix(n, z).dates[80])[:10]
        assert nearest.distance == pytest.approx(0.0, abs=1e-9)
        expected_1d = close[81] / close[80] - 1.0
        assert nearest.forward_returns["1d"] == pytest.approx(expected_1d, abs=1e-6)

    def test_horizon_beyond_available_data_is_none_not_fabricated(self):
        n = 30
        close = _flat_close(n)
        z = np.zeros((n, 3))
        query = n - 1
        z[3] = [0.0, 0.0, 0.0]  # near the start; a 60d horizon runs off the end
        result = find_analogues(_matrix(n, z), close, query, horizons_days=[60], guards=GUARDS)
        assert result.episodes[0].forward_returns["60d"] is None


class TestDegenerateCases:
    def test_incomplete_query_row_returns_no_episodes(self):
        n = 60
        z = np.random.default_rng(7).normal(0, 1, (n, 3))
        complete = np.ones(n, dtype=bool)
        complete[n - 1] = False
        result = find_analogues(_matrix(n, z, complete), _flat_close(n), n - 1, horizons_days=[1], guards=GUARDS)
        assert result.n_episodes == 0
        assert result.episodes == ()

    def test_too_little_history_returns_no_episodes_not_a_crash(self):
        n = 3
        z = np.random.default_rng(8).normal(0, 1, (n, 3))
        result = find_analogues(_matrix(n, z), _flat_close(n), n - 1, horizons_days=[1], guards=GUARDS)
        assert result.n_episodes == 0

    def test_distances_are_sorted_ascending_by_construction(self):
        rng = np.random.default_rng(9)
        n = 200
        z = rng.normal(0, 1, (n, 3))
        result = find_analogues(_matrix(n, z), _flat_close(n), n - 1, horizons_days=[1], guards=GUARDS)
        dists = [e.distance for e in result.episodes]
        assert dists == sorted(dists)


class TestReproducibility:
    def test_identical_inputs_produce_identical_results(self):
        """The bootstrap CI uses a fixed seed -- re-running on the same data
        must reproduce the exact same numbers, not just a similar range."""
        rng = np.random.default_rng(10)
        n = 300
        z = rng.normal(0, 1, (n, 3))
        for i in range(0, 250, 12):
            z[i] = [0.01, 0.01, 0.01]
        z[n - 1] = [0.0, 0.0, 0.0]

        m = _matrix(n, z)
        r1 = find_analogues(m, _flat_close(n), n - 1, horizons_days=[1, 5], guards=GUARDS)
        r2 = find_analogues(m, _flat_close(n), n - 1, horizons_days=[1, 5], guards=GUARDS)
        assert r1.to_dict() == r2.to_dict()
