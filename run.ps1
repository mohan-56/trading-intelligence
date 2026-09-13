# Crypto Intelligence -- one-command runner
#
#   .\run.ps1              start the app (default)
#   .\run.ps1 setup        create the venv and install dependencies
#   .\run.ps1 data         download crypto + context data (~20s, do this first)
#   .\run.ps1 all          data + start
#   .\run.ps1 test         run the test suite (offline, ~2s)

param([string]$Task = "serve")

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not $env:TI_DATA_ROOT) { $env:TI_DATA_ROOT = "C:/crypto-data" }
$py = ".\.venv\Scripts\python.exe"

function Need-Venv {
    if (-not (Test-Path $py)) {
        Write-Host "No virtual environment found. Run:  .\run.ps1 setup" -ForegroundColor Yellow
        exit 1
    }
}

switch ($Task) {
    "setup" {
        python -m venv .venv
        & $py -m pip install --quiet --upgrade pip
        & $py -m pip install --quiet -e ".[dev]"
        if (-not (Test-Path ".env")) { Copy-Item ".env.example" ".env" }
        Write-Host "`nReady. Next:  .\run.ps1 all`n" -ForegroundColor Green
    }
    "data" {
        Need-Venv
        Write-Host "`nDownloading BTC/ETH/SOL/BNB/XRP (spot + funding + OI) + DXY/VIX/US10Y context..." -ForegroundColor Cyan
        & $py scripts\backfill.py --years 15
    }
    "test" { Need-Venv; & $py -m pytest -q }
    "all" {
        Need-Venv
        & $py scripts\backfill.py --years 15
        & $PSCommandPath serve
    }
    "serve" {
        Need-Venv
        if (-not (Test-Path "$env:TI_DATA_ROOT/lake")) {
            Write-Host "`nNo data yet. Run:  .\run.ps1 data`n" -ForegroundColor Yellow
        }
        Write-Host "`n  Crypto Intelligence" -ForegroundColor Green
        Write-Host "  Dashboard : http://127.0.0.1:8000"
        Write-Host "  API docs  : http://127.0.0.1:8000/docs"
        Write-Host "  Ctrl+C to stop`n" -ForegroundColor DarkGray
        & $py -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --workers 1
    }
    default {
        Write-Host "Unknown task '$Task'. Use: setup | data | serve | all | test"
        exit 1
    }
}
