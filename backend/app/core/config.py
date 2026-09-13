"""Configuration. Everything env-driven; nothing hardcoded."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TI_", env_file=REPO_ROOT / ".env", env_file_encoding="utf-8", extra="ignore",
    )

    env: str = "development"
    log_level: str = "INFO"

    # Outside OneDrive: a Parquet lake in a synced folder sync-thrashes and
    # throws PermissionError mid-write.
    data_root: Path = Path("C:/crypto-data")

    api_host: str = "127.0.0.1"
    api_port: int = 8000

    num_threads: int = 4
    duckdb_memory_limit: str = "250MB"
    max_rss_mb: int = 500

    @field_validator("data_root", mode="after")
    @classmethod
    def _warn_onedrive(cls, v: Path) -> Path:
        if "onedrive" in str(v).lower():
            raise ValueError(f"TI_DATA_ROOT points inside OneDrive ({v}). Use e.g. C:/crypto-data.")
        return v

    @property
    def raw_dir(self) -> Path:
        return self.data_root / "raw"

    @property
    def lake_dir(self) -> Path:
        return self.data_root / "lake"

    @property
    def app_db_path(self) -> Path:
        return self.data_root / "app.sqlite"

    def ensure_dirs(self) -> None:
        for d in (self.data_root, self.raw_dir, self.lake_dir):
            d.mkdir(parents=True, exist_ok=True)


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@functools.lru_cache(maxsize=8)
def load_yaml_config(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def universe_config() -> dict[str, Any]:
    return load_yaml_config("universe.yaml")


def features_config() -> dict[str, Any]:
    return load_yaml_config("features.yaml")


def strategies_config() -> dict[str, Any]:
    return load_yaml_config("strategies.yaml")
