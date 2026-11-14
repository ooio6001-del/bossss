# SOLSTRATS paper trading bot (portable bundle)

**PAPER ONLY.** This bot never connects to a wallet, never signs a transaction and never places a real order.
Every buy and sell is simulated with virtual dollars on real, live market data. No private key is needed or used.

## What it does
Every hour, a few minutes after the hour closes, the bot:
1. Downloads the newest **complete** hourly candles from GeckoTerminal (free API, rate-limited with automatic backoff on HTTP 429)
   for the ~145 Solana meme-coin pools in `universe.json`. The still-forming current hour is ignored, same as the backtest.
2. Rebuilds each book's entry signal with the exact strategy code used in the backtests (`lib/`), applies the same exits
   (stop loss, trailing stop, time stop), and books trades with the same cost model as the backtest: DEX fee and slippage per
   venue, price impact, Solana base and priority fee per swap, token-account rent, and a cap of 5% of the bar's volume per fill.
   A buy fills at the next hourly open after the signal. Open positions are marked at the last close with exit costs included.
3. Writes its state, trade log, equity history and an HTML dashboard to `run/`.

It is restart-safe: each cycle replays every book deterministically from its start time, so a crash or reboot never
double-counts or loses a trade. Just run it again.

## Books (books.json)
| Book | Start | Sizing | Strategy |
|---|---|---|---|
| H7-07282 | $20 | 20% of equity per trade, up to 5 open | 10-hour closing low above the 336h average, after a 16%+ one-hour dump within 36h. Stop -33%, 22% trail after +65%, exit after 408h |
| H7-04028 | $20 | 50% of equity, up to 2 open | 8-hour closing low above the 336h average, after a 16%+ dump within 24h. Same exits |
| H7-07157 | $20 | 34% of equity, up to 3 open | 10-hour closing low above the 312h average, after a 16%+ dump within 36h. Same exits |
| three dollar buys (H6-01098 @ $3 start) | $3 | 28% of equity, up to 3 open | H6-01098: 8-hour closing low above the 120h average, after a 16%+ dump within 16h. Stop -30%, 20% trail after +60%, exit after 384h |
| H6-01098 native | $175 | 28% of equity, up to 3 open | H6-01098 |
| three dollar buys — $50 | $50 | fixed $3 per buy, long-only, as many open as cash covers | H6-01098 |

Edit `books.json` to add or change books. `frac` is the share of equity per buy. A negative `frac` means a fixed
dollar amount per buy, so `-3.0` is $3. `maxpos` is the maximum number of open positions. Delete `run/state.json` to
restart all books from zero with the new config.

**Daily pick:** if `run/active.json` exists, for example `{"id": "H7-07282", "date": "2026-11-14", "balance": 20}`, the bot
looks the strategy up in `strategies.csv`, then `data/robust_leaderboard.csv`, then `data/leaderboard.csv.gz` (all 80,000 hunted
variants) and starts a new book for that day's pick.

## Setup on a fresh machine (Python 3.10 or newer)
Mac / Linux:
```
unzip solstrats_bot.zip && cd solstrats_bot
./run.sh              # creates .venv, installs pinned requirements, starts in the background
./run.sh --foreground # same, but stays in the terminal
./run.sh --once       # one cycle, then exit (good first test)
kill $(cat run/trader.pid)   # stop
```
Windows: double-click `run.bat` or run it from a terminal. It runs in that window, and closing the window stops the bot.

No API key is required. `.env.example` lists optional settings. Copy it to `.env` if you need one. `HELIUS_API_KEY` is
optional, is not used for candles, and should stay empty unless you add on-chain checks.

## Files and logs
- `run/events.log`: every cycle, refresh stats and any errors (times in America/Chicago)
- `run/trader.out`: raw stdout/stderr, including rate-limit retries
- `run/state.json`: books, start times, current equity, open positions, closed trades
- `run/trades.csv`: every closed and open paper trade (entry/exit time and price, size, P&L, fees, exit reason)
- `run/equity.csv`: equity per book after every cycle
- `run/dashboard.html`: "LIVE PAPER" panel. Open it in a browser; it reloads every 5 minutes.
- `backtest_books.py [hours]`: backtest every book on the candles the bot has downloaded

## Data shipped in the bundle
- `data/price_cache/`: hourly candles for every pool up to Nov 13, 2026, 3 PM CST (May 18 onward). On first run they are
  copied to `run/data/`, and the bot downloads only the hours after the cache ends (up to 1000 per pool per request).
  Pools missing from the cache get their last ~41 days downloaded.
- `data/leaderboard.csv.gz`: all 80,000 hunted strategy variants (gzip, about 8 MB; the raw CSV is 41 MB)
- `data/robust_leaderboard.csv`: the 117 variants that passed rolling walk-forward
- `strategies.csv`: a slim list (robust rows plus the live picks) checked first

## Portability fixes (vs. the original box version)
- No absolute paths. Everything resolves from the bundle folder, or from `SOLSTRATS_HOME` / `SOLSTRATS_RUN_DIR` / `SOLSTRATS_DASHBOARD` if set.
- The strategy library (`wf_lib.py`, the hunter signal modules, the `solstrats` engine, cost model and HTTP client) is vendored in `lib/`.
- The bot uses a slim `strategies.csv` and `books.json` instead of requiring the 80K-row leaderboard. The full leaderboard is still shipped, gzipped, for daily-pick lookups.
- Starting candles come from the bundled cache first, then the bot fetches the rest itself. It no longer depends on another cache folder.
- The dashboard panel is written to a local `run/dashboard.html` instead of a shared dashboard file.
- Every file is under 50 MB, so the folder can be pushed to GitHub as is.

## Honest caveats
- Every backtest number is the best of many thousands of variants tested on the same six months, so expect live results to be worse.
  All three $20 picks lost 40% to 90% over the 30 days before Nov 13 even though the 14 days before that were profitable.
- H6-01098 with a $3 start lost about 95% in backtest because fixed Solana fees of about 6 cents per round trip dwarf buys under $1.
- Fills are modelled at the hourly open. A real bot would fill later and at a different price. Thin pools may not absorb the size.
- The watchlist is fixed (`universe.json`). Brand-new launches are not discovered.
- GeckoTerminal's free tier allows about 6 requests per minute from one IP, so a full hourly refresh takes about 15 to 25 minutes.
