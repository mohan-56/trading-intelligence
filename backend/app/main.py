"""FastAPI application factory. app/__init__ applies thread caps + TLS trust-
store routing before any numeric or networking library loads -- import order
matters, which is why `import app` comes before anything else below.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

import app as _app_pkg  # noqa: F401
from app.api.v1.routes import router as v1_router
from app.core.config import features_config, get_settings
from app.core.errors import AppError
from app.core.logging import configure_logging, get_logger
from app.domain.marketdata.service import MarketDataService
from app.domain.quant.features import load_matrix
from app.domain.symbols.registry import get_registry
from app.providers.http import close_client
from app.providers.registry import build_chains, build_providers
from app.storage.duck import close_connection, get_connection
from app.storage.lake import ParquetLake

log = get_logger("main")


@asynccontextmanager
async def lifespan(application: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level, json_output=settings.env == "production")
    settings.ensure_dirs()

    registry = get_registry()
    providers = build_providers()
    chains = build_chains(providers)
    lakes = {kind: ParquetLake(series_kind=kind) for kind in ("bars", "funding", "open_interest")}

    application.state.settings = settings
    application.state.registry = registry
    application.state.providers = providers
    application.state.chains = chains
    application.state.lakes = lakes
    application.state.marketdata = {
        "bars": MarketDataService(registry, chains["crypto"], lakes["bars"]),
        "funding": MarketDataService(registry, chains["funding"], lakes["funding"]),
        "open_interest": MarketDataService(registry, chains["open_interest"], lakes["open_interest"]),
    }

    # Read the precomputed state vector (~ms). Building it takes real time and
    # must never happen on a request. None is a valid state -- the API says
    # so and points at the script, rather than silently serving nothing.
    application.state.features = load_matrix(lakes["bars"].root, int(features_config().get("version", 0)))

    get_connection()
    log.info("startup_complete", symbols=len(registry), providers=sorted(providers),
             state_vector=("loaded" if application.state.features is not None else "NOT BUILT"),
             data_root=str(settings.data_root))
    try:
        yield
    finally:
        for provider in providers.values():
            await provider.aclose()
        await close_client()
        close_connection()
        log.info("shutdown_complete")


def create_app() -> FastAPI:
    application = FastAPI(
        title="Crypto Intelligence",
        version="0.1.0",
        description="Zero-cost crypto trading coach. Every figure carries provenance; nothing here is investment advice.",
        lifespan=lifespan,
    )

    @application.exception_handler(AppError)
    async def _domain_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(status_code=exc.http_status, content=exc.to_dict())

    application.include_router(v1_router)

    web_dir = Path(__file__).parent / "web"

    @application.get("/", include_in_schema=False)
    async def root() -> FileResponse:
        return FileResponse(web_dir / "index.html")

    return application


app = create_app()
