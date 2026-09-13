"""API integration tests -- against the real app, no network (empty lake)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture
def client():
    with TestClient(create_app()) as c:
        yield c


class TestSystemEndpoints:
    def test_health(self, client):
        assert client.get("/api/v1/health").json()["status"] == "ok"

    def test_status_reports_chains_and_lakes(self, client):
        body = client.get("/api/v1/status").json()
        assert "chains" in body and "lakes" in body
        assert set(body["chains"]) == {"crypto", "funding", "open_interest", "context"}

    def test_openapi_is_served(self, client):
        spec = client.get("/openapi.json").json()
        assert "/api/v1/bars/{ticker}" in spec["paths"]

    def test_dashboard_page_serves(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "Crypto Intelligence" in r.text


class TestReferenceEndpoints:
    def test_symbols_lists_the_universe(self, client):
        body = client.get("/api/v1/symbols").json()
        assert body["count"] == 8
        tickers = {s["ticker"] for s in body["symbols"]}
        assert {"BTCUSD", "ETHUSD", "DXY", "VIX", "US10Y"} <= tickers

    def test_coverage_reports_empty_lake_honestly(self, client):
        body = client.get("/api/v1/coverage").json()
        assert body["count"] == 8
        assert body["with_data"] == 0


class TestDataHonesty:
    def test_missing_data_returns_404_with_reason(self, client):
        r = client.get("/api/v1/bars/BTCUSD")
        assert r.status_code == 404
        body = r.json()
        assert body["error"]["code"] == "data_unavailable"
        assert body["error"]["detail"]["provenance"]["latency"] == "UNAVAILABLE"

    def test_unknown_symbol_is_404_not_500(self, client):
        r = client.get("/api/v1/bars/NOT_A_TICKER")
        assert r.status_code == 404

    def test_unknown_series_kind_is_404_with_options(self, client):
        r = client.get("/api/v1/bars/BTCUSD?kind=nonsense")
        assert r.status_code == 404
        assert "available" in r.json()["error"]["detail"]

    def test_bad_max_points_is_rejected(self, client):
        assert client.get("/api/v1/bars/BTCUSD", params={"max_points": 0}).status_code == 422
