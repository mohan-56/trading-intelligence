"""DuckDB session -- SQL over the Parquet lake, no server, no daemon."""

from __future__ import annotations

import threading

import duckdb

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger("duck")
_conn: duckdb.DuckDBPyConnection | None = None
_lock = threading.Lock()


def get_connection() -> duckdb.DuckDBPyConnection:
    global _conn
    if _conn is None:
        with _lock:
            if _conn is None:
                s = get_settings()
                conn = duckdb.connect(database=":memory:")
                conn.execute(f"SET memory_limit='{s.duckdb_memory_limit}'")
                conn.execute(f"SET threads TO {s.num_threads}")
                _conn = conn
                log.info("duckdb_ready", memory_limit=s.duckdb_memory_limit, threads=s.num_threads)
    return _conn


def query(sql: str, params: list | None = None) -> list[dict]:
    conn = get_connection()
    with _lock:
        rel = conn.execute(sql, params or [])
        cols = [d[0] for d in rel.description]
        rows = rel.fetchall()
    return [dict(zip(cols, row, strict=True)) for row in rows]


def close_connection() -> None:
    global _conn
    with _lock:
        if _conn is not None:
            _conn.close()
            _conn = None
