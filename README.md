# SOLSTRATS

Solana meme-coin strategy lab: historical backtests, walk-forward robust strategies, and a GitHub-styled paper-trading dashboard.

**Backtest and paper trading only. Not financial advice. No real trades, no wallet, no private keys. Meme coins routinely go to zero.**

Intended GitHub remote (do not put secrets in it): [https://github.com/ooio6001-del/bossss](https://github.com/ooio6001-del/bossss)

A separate public live-dashboard repo, [ooio6001-del/solstrats-dashboard](https://github.com/ooio6001-del/solstrats-dashboard), was copied **read-only** into `upstream/solstrats-dashboard/`. This bundle does not write to that repository.

## Headline strategy: H6-01098

Walk-forward champion from hunter batch 6 (`washout_gate`).

| | |
|---|---|
| **Entry** | First hour of an 8-bar closing low that is still above the 120h SMA, only if the token dumped 16%+ in a single hour within the last 16h |
| **Exit** | Stop −30%, 20% trailing stop after +60%, time exit after 384h |
| **Sizing** | 28% of balance per trade, up to 3 open; native start **$175** |
| **Walk-forward** | 94.1% of 17 rolling folds profitable (26 folds total); median OOS per fold +67.7% |
| **Full span at $175** | +556.7% ($175 → $1,149), 96 trades, max drawdown 54.5% |
| **Original 60/40 split** | IS +37.2%, OOS +400.0% (both windows profitable) |
| **Stress** | +47% at 2× costs, −94% at 3×, +164% if every entry fills one bar late, +272% without its best token; random-entry null p = 0.0066; 72% of 3,500 neighbouring configs profitable, median +70% |

It falls apart below about $30 because fixed fees (~$0.064 per round trip) eat the buys. Start-balance sweep on the same hourly history (ending equity):

| Start | End |
|---:|---:|
| $5 | $0.15 |
| $10 | $0.12 |
| $20 | $7.61 |
| $30 | $101 |
| $50 | $406 |
| $100 | $545 |
| $125 | $1,070 |
| $175 | $1,149 |
| $200 | $1,396 (and +95% over the last 30 days) |
| $280 | $1,511 |
| $500 | $1,614 |
| $1,000 | $1,491 |

Among those sizes, **$500** had the highest ending equity ($1,614). **$175** is the native size the walk-forward picked. **$30** is the lowest start that stayed clearly profitable. $1,000 ended lower than $500. Full params: `config/strategies/H6-01098.json` and `config/start_balance_H6-01098.json`.

Treat +557% as a selected, best-case number: most of the gain was in August, and the strategy lost money in May, June and July.

## What's in this repo

| Path | What |
|---|---|
| `solstrats/` | Python package: engine, costs, HTTP/providers, overlays, walk-forward, window engine, live paper runner |
| `tests/` | pytest suite |
| `*.py`, `run_all.sh`, `serve.py` | Lab scripts (backtest, overlays, live paper, report, date-window API) |
| `results/` | Selected CSV/JSON leaderboards, stress, windows, micro break-even (not the huge `.npz` sweeps) |
| `data/` | Token universe and on-chain snapshots (not the hourly OHLCV cache) |
| `hunt/` | Strategy hunter scripts, `robust_leaderboard.csv`, walk-forward / holdout evals, batch summaries, `ideas.md` |
| `live/` | $20 paper-trader scripts and state templates (no private keys) |
| `config/` | Strategy definitions (JSON), cost model, H6-01098 start-balance table, compact live snapshot |
| `dashboard.html` | Original standalone terminal (verbatim copy) |
| `dashboard_github.html` | GitHub-styled edition: same backtest boards plus live-repo settings, a Repository tab, and an Old vs New Compare tab |
| `upstream/solstrats-dashboard/` | Read-only copy of the public live dashboard (`data.json` from the `data` branch, plus `main` static files) |
| `.env.example` | Placeholder names only (`HELIUS_API_KEY=`, `BIRDEYE_API_KEY=`) |

## How to run

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # optional: add keys locally; never commit .env
python -m pytest -q tests
```

```bash
python serve.py            # http://127.0.0.1:8765/dashboard.html and /api/window?preset=24h
# or open the GitHub-styled file directly:
#   dashboard_github.html
```

Hourly OHLCV and the large sweep matrices are **not** in this bundle (see below), so a full re-sweep needs a data refetch (`fetch_history.py` / `run_all.sh`). The checked-in result tables and both HTML dashboards are enough to read the findings without refetching.

`serve.py` binds to `127.0.0.1:8765`.

## Public live dashboard (settings transferred into the GitHub Edition)

Copied read-only from [ooio6001-del/solstrats-dashboard](https://github.com/ooio6001-del/solstrats-dashboard).

- Branches: `data` (default, `data.json` only) and `main` (static GitHub Pages site: `index.html`, `data.json`, `solstrats-dashboard.html`, `README.md`, `.nojekyll`).
- There is **no** `package.json`, start script, or app server. The Pages site is already live at [https://ooio6001-del.github.io/solstrats-dashboard/](https://ooio6001-del.github.io/solstrats-dashboard/).
- `data.json` snapshot used here was exported **2026-11-13 17:36:01 CT** from the `data` branch (commit `521eee3`, message "data update 2026-11-13 17:34:12").

Transferred settings (see `config/live_snapshot.json` and the Live / Compare tabs):

- Paper only. Hunt start **$100**, min **$3** per buy (fixed $3–$100 or % of equity), up to **25** open; qualify at ≥20 closed trades, net equity > $100 after fees, profit factor > 1.2, not busted. Candidates: 8–19 closed trades. Ranking: net return % × trades/(trades+10).
- Grid shape `72x8x6x4x4x4x7` = 1,548,288 combos.
- Hunt population 88,000 (generation 131); 4,947 glitch-suspect combos excluded from the live leaderboard.

This live paper run is **not** the same data or the same strategies as the H6-01098 hourly backtest. See the Compare tab.

## Excluded from this bundle

Kept out so `solstrats_repo.tar.gz` stays under 24 MB, and so secrets never ship. None of these are required to read the dashboards or the checked-in result tables.

| Path (in the lab workspace, not here) | Why |
|---|---|
| `results/sweep.npz` (~1.2 GB), `results/trades.npz` (~84 MB), `results/overlay.npz` (~27 MB) | Huge binary matrices |
| `results_smoke/sweep.npz` (~841 MB) and other smoke binaries | Derived smoke run |
| `results/hunter_leaderboard.csv` (~32 MB), `hunt/leaderboard.csv` (~41 MB), `hunt/batch_1.csv`…`batch_8.csv` (4–7 MB each) | Full hunter grids; summaries and `robust_leaderboard.csv` are included |
| `results/sweep_meta.json`, `smart_search.json`, `report.json`, `overlay_meta.json` | Large derived JSON; key tables are included |
| `solstrats.zip` (~20 MB) | Earlier pack of the lab |
| `data/ohlcv/`, `data/cache/`, `data/live/`, `data/daily/`, `hunt/data_last48/`, `live/data/` | Hourly caches and live ticks (refetch to rebuild) |
| `live/screen20.pkl` (~3 MB) | Pickled screen |
| `__pycache__/`, `.numba_cache/`, `.pytest_cache/`, `.env` | Caches and secrets |

## Secrets

A scan of the lab, hunt, live, and upstream files found **no live API keys, tokens, or private keys**. Test fixtures use obvious placeholders (`k-123`, `secret123`). Provider snapshots may say `"helius": true` when a key was present in the *environment at runtime*; the key itself is not stored. Use `.env.example` as the only template.

## Method (lab backtest)

- Data: GeckoTerminal hourly OHLCV, 2026-05-18 → 2026-11-13 15:00 CT, 141 tokens (44 established, 97 launches). Public API caps history at 180 days.
- Split: first 60% of hours in-sample, last 40% out-of-sample. Rolling walk-forward is the headline robustness test.
- No look-ahead: signals on bar t, fills at bar t+1 open; stop before take-profit in the same bar.
- Multiple testing: millions of configs; the best number is partly luck. The Page states the caveat in full.
- Cost model: `config/costs.json`.

## License / use

Research and paper trading only. Not an offer to invest. Do not trade real size from these numbers without your own independent verification.
