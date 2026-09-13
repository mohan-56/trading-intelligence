"""Process bootstrap. MUST be imported before numpy, scipy, or duckdb.

Two jobs, both must happen before any numeric library loads:

1. Cap BLAS threads. numpy/OpenBLAS and DuckDB each try to grab every hardware
   thread. Three concurrent computations become a thread-count scramble and
   the UI stutters because the browser can't get a core. 4 threads leaves
   headroom on a normal laptop.

2. Route TLS verification through the OS trust store. Antivirus products with
   HTTPS scanning (Avast, Kaspersky, ESET) MITM every TLS connection and
   present their own certificate -- one that's in the Windows store but not
   in certifi's bundle, and that fails OpenSSL 3.x's strict validation
   ("Basic Constraints of CA cert not marked critical"). This was found by
   every single provider probe failing on first run. We never set
   verify=False -- full verification stays on, only the trust source changes.
"""

from __future__ import annotations

import os

_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

_applied = False
_trust_store_active = False


def apply_thread_caps(num_threads: int | None = None) -> int:
    global _applied
    if num_threads is None:
        num_threads = int(os.environ.get("TI_NUM_THREADS", "4"))
    num_threads = max(1, num_threads)
    for var in _THREAD_VARS:
        os.environ.setdefault(var, str(num_threads))
    _applied = True
    return num_threads


def use_system_trust_store() -> bool:
    global _trust_store_active
    try:
        import truststore

        truststore.inject_into_ssl()
        _trust_store_active = True
    except Exception:
        _trust_store_active = False
    return _trust_store_active


def caps_applied() -> bool:
    return _applied


def trust_store_active() -> bool:
    return _trust_store_active


apply_thread_caps()
use_system_trust_store()
