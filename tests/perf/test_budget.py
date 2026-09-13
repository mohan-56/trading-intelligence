"""Performance/policy budget, enforced."""

from __future__ import annotations

import importlib.util
import os

import psutil
import pytest

from app.core.config import get_settings

FORBIDDEN = ["torch", "transformers", "tensorflow", "jax", "ollama", "sentence_transformers"]


class TestNoLocalModels:
    @pytest.mark.parametrize("package", FORBIDDEN)
    def test_forbidden_package_is_not_installed(self, package):
        assert importlib.util.find_spec(package) is None, (
            f"'{package}' is installed. Policy: no local generative models -- "
            f"live crypto ticker is browser-direct WebSocket, no torch needed anywhere."
        )

    def test_no_gpu_dependency_declared(self):
        import tomllib

        from app.core.config import REPO_ROOT

        with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
            pyproject = tomllib.load(fh)
        declared = list(pyproject["project"].get("dependencies", []))
        for extras in pyproject["project"].get("optional-dependencies", {}).values():
            declared.extend(extras)
        for spec in declared:
            name = spec.split("[")[0].split(">")[0].split("=")[0].split("<")[0].strip().lower()
            assert name not in FORBIDDEN and not name.startswith("nvidia-")


class TestThreadCaps:
    def test_thread_env_vars_are_capped(self):
        from app.bootstrap import caps_applied

        assert caps_applied()
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            assert var in os.environ
            assert 1 <= int(os.environ[var]) <= 6

    def test_trust_store_is_active(self):
        from app.bootstrap import trust_store_active

        assert trust_store_active(), "TLS must route through the OS trust store (Avast/AV interception)"


class TestMemoryBudget:
    def test_import_footprint_is_small(self):
        rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        assert rss_mb < get_settings().max_rss_mb
