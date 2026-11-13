#!/usr/bin/env bash
# Full pipeline. Data fetch is slow on the free GeckoTerminal tier (~8-25 req/min): budget ~1h for a fresh fetch.
set -euo pipefail
cd "$(dirname "$0")"
[ -f .env ] && set -a && . ./.env && set +a
python3 -m pytest -q tests
[ -f data/universe.json ] || { echo "data/universe.json missing (ships with the zip)"; exit 1; }
python3 fetch_history.py 100        # hourly OHLCV -> data/ohlcv (cached; re-runs skip existing files)
python3 fetch_year.py               # 365-day daily context (CoinGecko)
python3 onchain.py                  # mint/freeze authority + holder concentration (Helius if keyed)
python3 run_backtest.py results     # brute-force grid (IS / OOS / full)
python3 run_overlays.py results     # agent & check overlays on every family
python3 search_smart.py 300 500     # Optuna TPE smart search (3 objectives)
python3 build_report.py results     # report.json + CSVs
python3 build_html.py results       # dashboard.html
echo "Live paper trading: python3 run_live.py 10 20 5,25,100,250,280  (then: python3 -m http.server 8000 and open http://localhost:8000/dashboard.html)"
