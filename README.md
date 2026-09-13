# Crypto Intelligence

A zero-cost crypto trading coach: regime detection, historical analogues, and
backtested strategies for BTC/ETH/SOL/BNB/XRP — with a genuinely real-time
price layer on top. See the [architecture doc](../architecture.html) for the
full design (layers, data flow, domain model, tech stack).

**Research and decision support. Not investment advice. No number on this
page is a prediction.**

## Status: Block A complete — data layer + live ticker

```
57 tests passing, offline, 2.4s          ruff clean
8/8 symbols ingested · 62,941 rows · 15 years (crypto) / max history (context)
Live WebSocket verified: 5/5 pairs, real-time, browser-direct to Binance
Backend RSS 114 MB (budget 500) · no torch, no GPU, no Docker
```

## Quick start

```powershell
.\run.ps1 setup   # venv + dependencies, once
.\run.ps1 all     # backfill data, then start the server
```

Open <http://127.0.0.1:8000> — live BTC/ETH/SOL/BNB/XRP prices streaming
directly from Binance, plus data coverage across all 4 series kinds (spot
bars, funding rate, open interest, macro context).

## What's here

| | |
|---|---|
| `backend/app/providers/crypto/binance.py` | Spot OHLCV + perpetual funding rate + open interest, all keyless |
| `backend/app/providers/macro/yahoo_context.py` | DXY / VIX / US10Y context signals |
| `backend/app/domain/marketdata/quality.py` | MAD-based anomaly detection, asset-aware negative-value rules |
| `backend/app/storage/lake.py` | Parquet lake, one file per symbol per series, mtime-cached reads |
| `backend/app/web/index.html` | Dashboard — live ticker (browser→Binance direct) + coverage |
| `scripts/backfill.py` | Full historical backfill: bars, funding, OI, context |

## What's next

Block B: regime detection + historical analogue search over this data.
Block C: backtester + risk engine + crypto-native strategies (funding-rate
carry, momentum, mean-reversion, breakout).
