"""Typed error hierarchy, mapped to HTTP at the API boundary."""

from __future__ import annotations


class AppError(Exception):
    http_status = 500
    code = "internal_error"

    def __init__(self, message: str, *, detail: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or {}

    def to_dict(self) -> dict:
        return {"error": {"code": self.code, "message": self.message, "detail": self.detail}}


class ConfigError(AppError):
    code = "config_error"


class SymbolNotFound(AppError):
    http_status = 404
    code = "symbol_not_found"


class DataUnavailable(AppError):
    """Requested data genuinely does not exist -- an honest 'we do not have it'."""

    http_status = 404
    code = "data_unavailable"


class ProviderError(AppError):
    http_status = 502
    code = "provider_error"

    def __init__(self, provider: str, message: str, *, detail: dict | None = None) -> None:
        super().__init__(f"[{provider}] {message}", detail=detail)
        self.provider = provider


class ProviderTimeout(ProviderError):
    code = "provider_timeout"


class ProviderRateLimited(ProviderError):
    http_status = 429
    code = "provider_rate_limited"


class DataQualityError(AppError):
    code = "data_quality_error"


class ValidationError(AppError):
    http_status = 400
    code = "validation_error"
