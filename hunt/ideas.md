# Strategy hunter — ideas log

This file records families already tried so later batches do not repeat them.
All tests use the local hourly GeckoTerminal cache in `/workspace/solstrats/data`
(2026-05-18 4:00 PM CDT through 2026-11-13 3:00 PM CST, 60/40 in-sample vs
out-of-sample split at 2026-09-03 2:00 AM CDT), the main engine's cost model
(DEX fee + slippage by venue, price impact, SOL tx fees, token-account rent),
and starting balances of $5, $25, $100, $250, and $280.

Main-grid families we do **not** re-grid here: breakout, ma_cross, rsi_revert,
vol_spike, dip_buy, moonshot, desk, fast_mc, graduation, bonding_curve,
narrative, flash_wick.

## Research inputs (batch 1)

- GraphDex, "Best Trading Strategies for Pump.fun Tokens in 2026": sniping,
  bonding-curve climb, graduation, **confirmed-winner first pullback**,
  narrative waves, smart-money mirror. We can test pullbacks and narrative
  lag on OHLCV; we cannot test wallet copy-trading (no wallet flow in the cache).
- MoonHydra, "How to Trade Solana Memecoins in 2026": launch snipe vs
  post-graduation momentum vs copy-trade vs **patient swing** on established
  names (MA pullbacks, opening-range, VWAP).
- MoonPay / AI sniper filter notes: stall exits, volume/buy-ratio filters.
  Buy-ratio and unique-wallet counts are not in this dataset; volume dry-up
  then expansion is the closest honest proxy.
- MadeOnSol early-buyer / smart-money patterns: useful live, not testable here
  without wallet history.
- Legal market-structure ideas only: theme lag, SOL-beta washout bounces,
  VWAP reclaim, failed breakdowns, climax-then-confirm. No bug exploits.

## Batch 1 families (10,000 variants)

80 entry parameterizations × 5 exits × 5 position-sizing rules × 5 balances.

1. **vwap_reclaim** — close back above the rolling VWAP after trading below it.
   Distinct from `ma_cross` (that is a fast/slow SMA cross, not a VWAP reclaim).
2. **rel_strength** — token beats SOL over 24h or 72h, optionally only after a
   5% pullback from the 24h high. The main grid never uses SOL as a relative
   benchmark in an entry rule (`regime_sol_uptrend` is an overlay, not this).
3. **pullback_ma** — bounce back above a fast SMA while still above a slow
   trend SMA. Distinct from `ma_cross` (cross event) and `dip_buy` (drawdown
   from a high, no trend MA).
4. **failed_breakdown** — same-bar spring: the hour's low breaks the prior
   N-hour low but the close reclaims it. Distinct from `flash_wick` (signals
   on a close-to-close dump, no reclaim requirement) and `dip_buy`.
5. **theme_laggard** — daily, buy names in the *hottest* theme that are
   *trailing* the theme. Opposite of main-grid `narrative`, which buys tokens
   that are already up with the theme.
6. **post_run_pullback** — first bounce after a 40% or 100% run over 24h/72h
   that has given back 10–20% from the run high. GraphDex "confirmed winner
   catch"; distinct from generic `dip_buy` because a prior explosive run is
   required.
7. **climax_reversal** — wait for a green confirmation hour *after* a high-
   volume dump. Distinct from `flash_wick`, which signals on the dump bar
   itself and fills the next open without a reversal confirm.
8. **sol_risk_on** — buy a green meme hour after SOL has dropped 3–6% over
   6h or 24h, optionally only if the token held up better than SOL.
9. **dryup_expansion** — several hours of below-average volume, then a green
   expansion bar. Distinct from `vol_spike` (no dry-up prerequisite).
10. **session_orb** — close through the first 2h or 4h UTC opening-range high,
    all tokens or established-only. The main grid has no session/time-of-day
    rule.

Exits swept: scalp (+5% / −30% / 24h), standard (+50% / −20% / 168h), wide
(+100% / −35% / 240h), uncapped trail (no TP, −35% stop, 25% trail after +50%,
720h), tight trail (no TP, −15% stop, 15% trail after +25%, 240h).

Sizing: 8%/12, 20%/5, 50%/2, 100%/1, 5%/20 (fraction of equity / max positions).

## Not tested in batch 1 (candidates for later batches)

- Stochastic / Williams %R oversold in an uptrend
- Bollinger or Keltner squeeze-then-break
- Inside-bar / engulfing-only patterns (close cousins of failed_breakdown)
- Consecutive green-stack momentum
- ATR channel break
- Cross-DEX or pool-lag spreads (dataset has one pool per token, so a real
  cross-venue lag test is not possible yet)
- Wallet-level smart-money or team-funding copy rules (no wallet graph here)
- Hour-of-day filters layered on existing families
- Consecutive-loss cooldown / time-of-week filters
- Partial scale-out schedules (engine is all-in/all-out per position)

## Batch 1 results

Ran 2026-11-13, about 3:22–3:23 PM CST. Exactly 10,000 variants. 9,970 had at least
one out-of-sample trade; 704 were profitable out of sample; only 97 were profitable
in both the in-sample and out-of-sample windows. Data: 4,297 hourly bars, 119 tokens
(including the SOL benchmark), costs as in the main engine.

Family hit rates (OOS profitable / 1,000 configs; configs profitable in both windows):

- climax_reversal: 483 OOS-profitable, 54 both — highest hit rate, but signals are
  thin (516 raw signals across 8 param sets; many configs have 0–3 OOS trades).
- sol_risk_on: 99 OOS-profitable, 5 both.
- vwap_reclaim: 34 OOS-profitable, 9 both. Best *validated* family.
- rel_strength: 33 OOS-profitable, 12 both. Raw OOS leader, but the huge numbers
  failed in-sample (near-wipe) and over the full period.
- failed_breakdown: 17 OOS-profitable, 2 both. Second raw OOS profit also failed IS.
- post_run_pullback: 15 / 7 both, small profits (best OOS +$26.62).
- dryup_expansion: 11 / 7 both, best OOS +$49.90.
- session_orb: 6 / 0 both — no config made money in both windows.
- pullback_ma: 5 / 0 both — same, no walk-forward confirmation.
- theme_laggard: 1 / 1 both, leftover +$0.16. Effectively a miss on this universe.

Raw top 5 by OOS profit (requested ranking) are listed in batch_1_summary.json.
Important: #1–#4 all lost money in-sample (the relative-strength all-in variant
lost 99% in-sample and finished the *full* period at −99.9%). Those OOS profits
use a reset starting balance on the later window, not a continuous account. They
should not be treated as a discovered edge.

The best variant that was profitable in *both* windows, with at least 5 trades
on each side, is H1-00094: 12h VWAP reclaim, no volume filter, −35% stop, 25%
trail after +50%, 720h time stop, 100% of a $250 balance, 1 position. OOS
+$538.99 (215.6% , 14 trades, 71.4% win, 38.0% max DD). IS +9.1% on 7 trades.
Full-period +236.7%. A 50%-sizing cousin (H1-00090, $280 start) had a sturdier
IS (+17.5%, 12 trades) and OOS +$420.08 (26 trades, 65.4% win).

Compared with the main hourly grid's best OOS profit (flash_wick G26453-S3,
+$225.31 on $225, 14 OOS trades, IS only +1.3%): several raw hunt variants beat
it, and the validated VWAP reclaim also beats it on OOS profit. The thin
moonshot max-return print (+$565.77 on $15, 2 OOS trades, negative IS) is still
slightly ahead of the validated VWAP number, but that moonshot result is not a
serious comparator.

## Do not repeat in later batches

All 10 families and the exact 80 entry parameterizations, 5 exits, 5 sizing
rules, and 5 balances above. Later batches should add new families or new
parameter regions (for example squeeze-breaks, hour-of-day overlays, or
tighter VWAP windows other than 12/24/48/72h) rather than re-running this grid.

## Research inputs (batch 2)

- XBankang, "Daily Memecoin Trading Strategies": US-morning (13:00–16:00 UTC) confluence
  on VWAP reclaims; US-session opening-range breakouts (not the UTC-midnight ORB from
  batch 1); mean-reversion fades after blow-off wicks.
- LiveVolatile, "Solana Volatility Scalping Strategy 2026": ATR/low-vol squeeze then a
  volume-confirmed close outside the range (Bollinger / Keltner-style).
- Secuora / CoinSkeptic / Tickfloor squeeze backtests: a squeeze flags *when*, not
  *direction*; first close beyond the band is often a false poke. We therefore test a
  retest-of-the-band variant, not only the first poke.
- DogeKing meme-coin TA 2026: volume-confirmed breaks; resistance-turned-support flips
  (inside-bar / NR7 are the clean OHLCV versions we can run here).
- StrategyVerdict Williams %R study: standalone oscillators can post high win rates and
  still lose money. We only test stochastic *crosses up through oversold while price is
  above a long moving average*, not a naked oscillator.
- Batch 1 follow-up: refine the three families that actually printed in both windows
  (VWAP reclaim, climax reversal, sol_risk_on) in *new* parameter regions, with smaller
  clips and more concurrent seats instead of the batch 1 all-in / 50% sizing.

## Batch 2 families (10,000 new variants)

80 entry parameterizations × 5 *new* exits × 5 *new* diversified sizing rules × the
same five balances. No reuse of batch 1's entry × exit × sizing grid.

Refinements (new parameter regions only):

1. **vwap_reclaim** — windows 6, 8, 16, 36, 96, 168h (not 12/24/48/72); volume
   multipliers 1× / 1.5× / 3× (not 0 or 2); optional US-morning 13:00–16:00 UTC filter.
2. **climax_reversal** — dump cuts 10%, 12%, 18%, 20%, 30% (not 8/15/25/40); volume
   1.5× / 3× / 5× (not 2 or 4); some variants require the confirmation close to also
   clear the dump bar's open.
3. **sol_risk_on** — SOL dumps of 4%, 5%, 8%, 10% over 3h / 12h / 36h / 48h (not the
   batch 1 3–6% × 6h/24h grid), with and without a "held up better than SOL" filter.

Fresh families:

4. **stoch_uptrend** — 9h or 14h stochastic %K crosses up through 20 or 30 while price
   is above its 72h or 168h average.
5. **bb_squeeze_break** — close above the upper 20h or 40h Bollinger band (2σ) after a
   48h or 96h width squeeze; first poke *or* a later retest of the band.
6. **nr7_break** — close above the high of the narrowest-range bar of the last 5 or 7
   hours, optional 2× volume and +2% breakout-hour filter.
7. **inside_break** — close above the mother-bar high after 1 or 2 inside bars,
   optional volume filter and "mother bar was red".
8. **us_session_orb** — close through the first 1h or 2h US-morning (13:00 UTC)
   opening-range high. Distinct from batch 1's UTC-midnight session_orb.

Exits swept (all new): +15% / −12% / 48h; +35% / −18% with a 10% trail after +20% /
120h; +75% / −25% / 336h; +200% / −40% with a 30% trail after +100% / 480h; no TP,
−22% stop, 20% trail after +40% / 360h.

Sizing (all new, diversified): 4%/25, 6%/15, 10%/10, 15%/6, 25%/4.

## Batch 2 results

Ran 2026-11-13, about 3:25–3:26 PM CST. Exactly 10,000 variants. Data this run: 4,297
hourly bars, 123 tokens (including SOL; a few more files had landed since batch 1),
same 60/40 split and cost model. 9,590 configs had at least one OOS trade; 1,374 were
OOS-profitable; **303 were profitable in both windows** (97 in batch 1); 293 of those
had at least 5 trades on each side; 1,071 OOS-profitable names were in-sample losers
and are flagged as flukes.

Primary ranking is both-window profitability, then OOS profit.

Family hit rates (OOS-profitable / both-window / OOS-only flukes):

- climax_reversal: 1,063 / 241 / 822 — again the highest both-window count, and with
  the new 10–18% dump cuts plus 4–15% position sizes the *validated* names now have
  dozens of trades, not one-trade lottery tickets.
- bb_squeeze_break: 110 / 34 / 76 — the only fresh family with a real both-window
  cluster. Best validated is a *retest* (not first-poke) 40h/96h squeeze on a $25
  account, +$34.02 OOS (136%).
- stoch_uptrend: 69 / 6 / 63 — raw OOS leader ($651) is a fluke (lost 19% in-sample).
  Validated best is only +$80.98.
- sol_risk_on: 47 / 8 / 39 — refinements stayed modest. Best both-window with ≥5
  trades: H2-03842, $25 start, +$30.45 OOS (122%) and +52.5% IS, 22/32 trades, full
  period +160%.
- vwap_reclaim: 38 / 8 / 30 — new windows and US-morning filter plus smaller clips did
  **not** beat batch 1's 12h all-in reclaim. Best both-window here is only +$24.81, and
  that name's *continuous* full-period return is negative, so even the both-window
  print is shaky.
- nr7_break: 37 / 2 / 35 — raw OOS +$614 is a fluke (IS −4.7%, full period −46.5%).
  The one sturdy validated cousin (H2-07950) made +$104.76 OOS on $280 (26/33 trades).
- inside_break: 8 / 4 / 4 — small, best both-window +$48.73, but full period negative.
- us_session_orb: 2 / 0 / 2 — another session-ORB miss, same story as batch 1's UTC
  midnight ORB. No config made money in both windows.

Raw top 5 by OOS profit are **all flukes** (stoch and NR7). The raw leader, H2-05225
(9h stochastic cross of 20 above the 168h average, +$651.17 on $280, 31 OOS trades),
lost $54.51 (−19.5%) in-sample. Do not treat those as edges.

Best both-window by OOS profit is H2-03375: 30% climax dump at 1.5× volume, −22% stop,
20% trail after +40%, 360h, 25% of a $280 balance, 4 seats. OOS +$194.98 (69.6%, 6
trades, 50% win, 12.3% DD), IS +$17.66 (6.3%, only 3 trades). Full period +80.2%. It
does *not* beat the main hourly-grid flash_wick best (+$225.31) or batch 1's validated
12h VWAP reclaim H1-00094 (+$538.99). In-sample is also thin (3 trades).

The sturdiest both-window names (also ≥5 trades each side) are diversified climax
reversals, especially:

- H2-02483: 12% dump / 1.5× vol, same mid-trail exit, 6% of $100, up to 15 open.
  OOS +$133.38 (133.4%, 60 trades, 43.3% win, 13.0% DD, PF 3.22). IS +37.8% on 63
  trades. Full period +100.4% ($200.44), 122 trades, 15.7% DD.
- H2-02108: 10% dump / 1.5× vol, same exit and 6%/$100/15 sizing. OOS +$128.33 (73
  trades, 42.5% win, 10.6% DD). IS +89.2% on 77 trades (41.6% win). Full period
  +181.8% ($281.79), 145 trades, 12.4% DD. Win rate is almost identical in both
  windows — the most consistent walk-forward print in this batch.

Diversified sizing (4–15% clips, 6–25 seats) is what made the climax family look
honest this time: smaller dollars than batch 1's all-in lottery tickets, but far
more trades and matching IS/OOS behaviour.

## Do not repeat in later batches

Batch 1's 10 families and exact 80×5×5×5 grid, **plus** this batch's 8 families /
parameter regions, 5 exits, and 5 sizing rules. Later batches should not rerun
6/8/16/36/96/168h VWAP, these climax dump/vol cuts, these SOL-dump windows, this
stochastic grid, this BB squeeze grid, NR5/NR7, these inside-bar rules, or the
13:00 UTC US ORB grid.

Still open: Keltner / ATR-channel breaks, hour-of-day overlays on climax (not
VWAP), consecutive-loss cooldowns, resistance-flip after a broken high, z-score
mean reversion in a low-vol regime, and any future dataset that actually has
wallet flow or two venues per token (still not testable here). Partial scale-outs
still cannot be run in the all-in/all-out engine.

## Research inputs (batch 3)

- FloofyJin MarketStrategies / regime-gated confluence notes: require at least two
  independent conditions; in a range, pair VWAP with a recent shock; do not stack
  two redundant mean-reversion rules. Dual confirmation cuts most signals and
  keeps higher-conviction setups.
- SpreadAtlas indicator study (2026): Keltner and Bollinger *breakouts* fare better
  than oversold fades on 1h/4h bars after costs; we test a Keltner close above
  EMA+k×ATR, optionally after ATR compression.
- Multi-strategy ensemble bots (MultiStratEngine, Alcyone): one shared account
  combining signals, plus z-score mean reversion in a low-vol regime.
- Batch 2 "still open" list: Keltner/ATR-channel breaks, hour-of-day overlays on
  climax (not VWAP), resistance-flip after a broken high, z-score revert in low
  vol, consecutive green-stack momentum.
- Explicit user ask: ensembles of the validated families (VWAP + climax, and
  friends) sharing one account.

## Batch 3 families (10,000 new variants)

80 entry parameterizations × 5 *new* exits × 5 sizing rules (all-in 100%/1, reused
only with new exits/filters, plus new diversified clips 3%/30, 7%/14, 12%/8, 18%/5)
× the same five balances. No reuse of the batch 1 or batch 2 entry × exit × sizing
grids.

Refinements of the leaders:

1. **vwap_reclaim** — locked to the 12h window that produced H1-00094, but every
   variant adds a new filter: close in the upper half of the hour, minimum reclaim
   distance, SOL above its 24h average, a 4% cap above VWAP, 0.5×/1.2× volume, or
   a one-hour delayed confirmation. Not a rerun of naked 12h / vm=0 or 2.
2. **climax_reversal** — new dump/vol cuts (9/11/14/16/22/28% × 1.2/2.5/6×, not
   the batch 1 or 2 grids) plus hour-of-day overlays (US morning, Asia 00:00–08:00
   UTC, EU 07:00–12:00 UTC).
3. **bb_squeeze_break** — retest only, on new lengths (24/30/50/60h), 1.5σ or 2.5σ,
   72/120/144h squeezes. Not the batch 2 20/40h × 2σ × 48/96h grid.

Ensembles (new family, one shared account, one position per token):

4. **ensemble** — OR unions of the validated-style components (12h VWAP, 10%/12%/18%
   climax confirms, 40h/2σ/96h Bollinger retest) so those signals share a book; and
   confluence gates (climax only while above 12h VWAP; VWAP reclaim only after a
   recent 10% climax dump; BB retest only above VWAP; OR union skipped when SOL is
   down 5%+ over 24h). The component signals are not rerun as standalones.

Fresh families:

5. **keltner_break** — close above the 20h or 40h Keltner upper band (EMA + 1.5× or
   2.5× ATR), optionally only after ATR is below 70% of its 48h average.
6. **zscore_revert** — green bounce when the 24h or 48h close z-score is at most
   −1.5 or −2.5, optionally only in a low-vol regime (14h ATR below its 48h average).
7. **resist_flip** — bounce that holds a broken 24h or 72h high as support.
8. **green_stack** — close of the 3rd, 5th, or 6th consecutive green hour, optional
   rising volume and a +2% per-hour filter.

Exits swept (all new): +20% / −25% / 96h; +60% / −30% with an 18% trail after +30%
/ 288h; +150% / −28% / 192h; no TP, −40% stop, 28% trail after +60% / 600h; +40% /
−16% with a 12% trail after +15%, 45% crash exit, 72h.

## Batch 3 results

Ran 2026-11-13, about 3:29–3:30 PM CST. Exactly 10,000 variants. Data this run:
4,297 hourly bars, 126 tokens (including SOL), same split and cost model. 9,755
configs had at least one OOS trade; 1,232 were OOS-profitable; **377 were
profitable in both windows** (97 in batch 1, 303 in batch 2); 324 of those had at
least 5 trades on each side; 855 OOS winners were in-sample losers (flukes).

`leaderboard.csv` now has 30,000 rows and a `both_windows` column (`true`/`false`).
All prior rows were backfilled. 777 rows are `true` (97+303+377).

Primary ranking is both-window profitability, then OOS profit.

Family hit rates (OOS-profitable / both-window / OOS-only flukes):

- climax_reversal: 641 / 139 / 502 — still the densest both-window family. The raw
  both-window dollar leader is an all-in 22% dump with only 2 in-sample trades
  (thin). The sturdiest diversified cousin is a 16%/2.5× dump at 7% of $280.
- ensemble: 205 / 90 / 115 — the important new family. Simple OR unions mostly
  look like a milder climax book. The *confluence* "12h VWAP reclaim only after a
  recent climax dump" is the best walk-forward idea in this batch: strong
  in-sample *and* out-of-sample, many trades, positive full period.
- keltner_break: 142 / 54 / 88 — best fresh family by both-window count. Best
  validated: 20h / 2.5×ATR after a squeeze, +$94.69 on $100 (22/32 trades).
- zscore_revert: 86 / 36 / 50 — works in the low-vol regime; best both-window
  all-in on $280 is +$61.22 OOS / +$197.67 IS (only 8/10 trades, 47% OOS DD).
- bb_squeeze_break: 67 / 40 / 27 — new retest lengths did not improve on the
  batch 2 40h/96h retest; best both-window here is only +$7.10.
- green_stack: 53 / 10 / 43 — 3 green hours each up 2%+, 7% of $280, +$117.78 OOS
  on 80 trades, but IS only +2.7%.
- vwap_reclaim: 29 / 8 / 21 — 12h filters plus new exits/sizing did **not** beat
  H1-00094. The delayed-confirm all-in (+$287.72 OOS) is a fluke (IS −60%, full
  period −94.5%). The best both-window 12h refine is a tight 3% clip, +$38.23 OOS
  / +43.1% IS, 147/192 trades.
- resist_flip: 9 / 0 / 9 — no config made money in both windows. A miss.

Raw top 5 by OOS profit are **all flukes**, all all-in climax variants. The raw
leader H3-02598 made +$843.51 on $100 out of sample and lost $79.93 (−79.9%)
in-sample (full period of a close cousin can look fine or can wipe; this one
still has a negative IS). Do not treat those as edges.

Best both-window by OOS profit is H3-02625: 22% climax dump at 2.5× volume, +40%
TP / −16% stop / 12% trail after +15% / 45% crash / 72h, 100% of $280, 1 seat.
OOS +$294.71 (105.3%, 6 trades, 66.7% win, 19.2% DD). IS +$22.50 (8.0%) on only
**2 trades**. Full period +119.5%. It beats the main hourly-grid flash_wick best
(+$225.31) on OOS profit, but the in-sample side is too thin to trust, and it
does **not** beat batch 1's validated 12h VWAP reclaim H1-00094 (+$538.99), which
is still the cumulative both-window leader.

The sturdiest both-window names (≥5 trades each side, full period also up):

- H3-02335 (diversified climax): 16% dump / 2.5× vol, −40% stop, 28% trail after
  +60%, 600h, 7% of $280, 14 seats. OOS +$163.01 (58.2%, 26 trades, 50% win,
  11.7% DD). IS +2.8% on 21 trades. Full period +54.3%. Honest but a weak IS.
- H3-04834 (OR ensemble): 18%/3× climax OR 40h Bollinger retest, same 7%/$250/14
  trail. OOS +$130.79 (31/34 trades). IS only +4.0%. Looks like a milder climax
  book, not a new edge.
- H3-05689 (confluence, the quality find): 12h VWAP reclaim only if a 10%+/1.5×
  climax dump occurred in the last 12h; +150% TP / −28% stop / 192h; 12% of $250,
  8 seats. OOS +$112.41 (45.0%, 49 trades, 28.6% win, 33.5% DD). IS +$255.85
  (+102.3%) on 34 trades. Full period +122.6% ($556.61), 75 trades. A 6h version
  (H3-05569) has an even stronger full-period +161.8% with IS +96.3% and OOS
  +$84.32. This is the first ensemble that clearly helps walk-forward, because
  the VWAP entry is gated by a recent washout rather than OR-merged with it.

Keltner after a squeeze (H3-06468, +$94.69 OOS on $100, 22/32 trades, full +93%)
is the best *new standalone* family. Resistance flips failed walk-forward.

## Do not repeat in later batches

Everything already listed for batches 1 and 2, **plus** this batch's 12h VWAP
filters, these climax dump/vol/session cuts, these BB retest lengths, this
ensemble/confluence set, the Keltner 20/40h × 1.5/2.5 × squeeze grid, this
z-score grid, these resistance-flip lookbacks, this green-stack grid, and these
five exits and four new sizing rules (3%/30, 7%/14, 12%/8, 18%/5).

Still open: consecutive-loss cooldowns, ATR-sized stops (the engine uses percent
stops), other confluence pairs (e.g. Keltner + climax), hour-of-day on ensembles,
and any future dataset with wallet flow or two venues per token. Partial
scale-outs still cannot be run in the all-in/all-out engine.

## Research inputs (batch 4)

- SuperTrend regime-confluence writeups (DefinedEdge and similar): flip the ATR
  trailing band only with a washout or ADX gate; skip range regimes.
- Donchian breakout literature (AegisQuant and others): pair a channel break with
  an ADX trend filter or a pullback to the mid, rather than taking every N-bar high
  (the main grid already tests plain breakouts).
- Solana Day Trader / Williams VIX Fix washout notes: a WVF extreme plus a
  confirmation bounce is a structural-flush entry, distinct from a raw flash wick.
- 3-bar fair-value-gap reclaim as a simple OHLCV imbalance (used here as a fresh
  family and as a washout-gated confluence).
- Batch 3 result to expand: H3-05689 / H3-05569, a 12h VWAP reclaim *gated by a
  recent climax dump*. New pairs include Keltner breaks gated by a token washout,
  Bollinger retests gated by SOL risk-on, SuperTrend flips after dumps, VWAP after
  WVF extremes, and US-morning hour filters on those gates. The exact 10%+/1.5× ×
  6h/12h VWAP-after-climax entries are not rerun.

## Batch 4 families (10,000 new variants) + H1-00094 walk-forward

80 entry parameterizations × 5 *new* exits × 5 *new* sizing rules (2%/40, 9%/11,
11%/9, 14%/7, 16%/6) × the same five balances. Walk-forward folds of H1-00094 are
saved separately and are *not* part of the 10,000.

Confluence (40 entries, one shared account, entry AND gate):

1. **confluence** — 12h VWAP, 14h/30h Keltner (2× or 3× ATR, some after a squeeze),
   new Bollinger retests (28h/1.8σ/84h, 36h/2σ/80h, 45h/2.2σ/100h), SuperTrend(10,3)
   or (14,2), 36h z-score ≤ −2, 4-green stack, or a 24h FVG reclaim, each gated by a
   token hourly dump, a 14%/16%/22% climax confirm, a SOL window dump (with or
   without the token holding up), a WVF extreme, a 24h spring, or an ADX<25 range
   filter. A few are further limited to 13:00–16:00 UTC.

Fresh families (8 each):

2. **supertrend** — SuperTrend(7/10/14, 2 or 3) bullish flip, optionally only when
   ADX is at least 20 or 25.
3. **adx_donchian** — close above a 36/48/96/120h Donchian high only when 14h ADX is
   at least 20 or 25 (not the main-grid 24/72/168h unfiltered breakout).
4. **fvg_reclaim** — reclaim of a bullish 3-bar fair-value gap formed in the last
   12–72h.
5. **wvf_washout** — first green hour after a Williams VIX Fix(14 or 22) reading of
   at least 80–90%.
6. **donchian_mid** — bounce back above a 24–96h Donchian mid after a pullback,
   while the mid has been rising.

Exits swept (all new): +12% / −14% / 36h; +45% / −22% with a 14% trail after +25%
/ 144h; +90% / −32% / 216h; no TP, −28% stop, 22% trail after +45% / 504h; +120% /
−18% with a 16% trail after +50%, 50% crash exit, 400h.

## H1-00094 walk-forward (not part of the 10,000)

Rules retested: 12h VWAP reclaim, no volume filter, −35% stop, 25% trail after +50%,
720h, 100% of $250, 1 position. 26 folds on the current cache (4,297 hours, 130
tradable tokens vs 118 when the leader was first printed). Results are in
`walkforward_H1-00094.csv` and `.json`.

On the original 60/40 split (2026-09-03 2:00 AM CDT) with current data: IS still
+9.1% on 7 trades (matches the published print). OOS is still both-windows
profitable but a bit weaker: +$512.76 (205.1%, 10 trades) vs the published +$538.99
(215.6%, 14 trades). The full-period path still matches exactly (+236.7% to
$841.70, 21 trades, 57.5% DD). The universe has grown and some bars may have been
revised; this is not a bit-exact replay.

It does **not** hold up under other splits:

- All 26 folds: only 8 are OOS-profitable (31%) and only 3 are profitable in both
  windows (12%). Median OOS return is −20.0%. The original +205% sits at the 88th
  percentile and well above the other-fold median (−22.6%).
- Anchored 40/50/60/70/80% splits: 2 of 5 both-windows. The 40% and 50% splits lose
  on both sides (OOS −78% and −67%). The 70% split makes +131% in-sample then
  loses 59% after 2026-09-20. The 80% split is the other both-window name
  (+119% OOS on 5 trades).
- Calendar month-end splits: 0 of 4 both-windows. The split after August is an
  OOS-only fluke (+279% after an IS loss of 45%) — the same late-August / early-
  September cluster that flatters the original 60/40 window.
- Rolling windows (60d/30d, 45d/21d, 70d/35d): 17 folds, 5 OOS-profitable, **only
  1 both-windows**. That one both-window roll starts OOS on 2026-08-31 and prints
  +304%. Most other rolls lose 10–40%.

Honest read: H1-00094's big 60/40 OOS number is concentrated in a short late-
summer window, not a stable edge across split dates. It remains the cumulative
leaderboard's top both-window print for that one split, but it should not be
treated as a robust walk-forward winner.

## Batch 4 results

Ran 2026-11-13, about 3:32–3:33 PM CST. Exactly 10,000 variants. Data this run:
4,297 hourly bars, 131 tokens including SOL, same 60/40 split and cost model.
8,750 configs had at least one OOS trade; 702 were OOS-profitable; **379 were
profitable in both windows**; 301 of those had at least 5 trades on each side;
323 OOS winners were in-sample losers. `leaderboard.csv` now has 40,000 rows and
1,156 `both_windows=true` (777 + 379).

Family hit rates (OOS-profitable / both-window / OOS-only flukes):

- confluence: 464 / 211 / 253 — the densest both-window family, and the raw OOS
  leader of this family is also both-windows (no hidden fluke at the top).
  Token-washout gates work; SOL risk-on gates mostly do not.
  - Best: 14h Keltner squeeze break gated by a 12%+ token dump in the last 24h
    (H4-01464), +$109.48 OOS / +$141.16 IS, low drawdown.
  - 14h Keltner break gated by a 16%+/2.5× climax in the last 24h: 69 both-window
    configs, best +$93.61.
  - 12h VWAP reclaim gated by a 12%+ dump in the last 24h (a cousin of H3-05689
    with a raw hourly-dump gate instead of the 10%/1.5× climax confirm): works at
    tiny 2% clips, +$87.62 OOS on 189 trades, but IS only +3.9%.
  - VWAP gated by a WVF(14) ≥85% in the last 6h: 19 both-window, best +$44.30.
  - **Bollinger retest gated by SOL risk-on: 0 both-window configs.** VWAP gated
    by a SOL dump is also 0 both-window. The gate that transfers is a *token*
    washout, not a SOL-beta washout.
- wvf_washout: 183 / 152 / 31 — highest both-window *rate* of the fresh families,
  but modest dollars. Best: WVF(22) ≥80%, +$43.38 on $100 (15/16 trades, IS +17%,
  full +52%).
- fvg_reclaim: 30 / 5 / 25 — the raw OOS top 3 of the whole batch are FVG flukes
  (best +$157.29 OOS, IS −63%, full period −71%). Validated FVG tops out at +$20.
- donchian_mid: 15 / 6 / 9 — small, best +$18.96 on $280, many trades, ~7–8% in
  both windows.
- supertrend: 10 / 5 / 5 — standalone flips are weak; they only look useful as a
  washout-gated confluence. Best standalone both-window +$25.89, but the full
  period is slightly negative.
- adx_donchian: 0 / 0 / 0 — a complete miss on this universe. No OOS-profitable
  config.

Raw top 5 by OOS profit: four FVG flukes, then the validated Keltner-washout
confluence (H4-01464). Do not treat the FVG prints as edges.

Best both-window by OOS profit is H4-01464: 14h Keltner (EMA+2×ATR) squeeze break
only after a 12%+ token dump in the last 24h; −28% stop, 22% trail after +45%,
504h; 11% of $250, up to 9 open. OOS +$109.48 (43.8%, 17 trades, 41.2% win, 8.4%
DD, PF 2.30). IS +$141.16 (+56.5%) on 12 trades (41.7% win). Full period +59.6%
($398.93), 28 trades, 20.3% DD. Win rates nearly match across windows. It does
**not** beat H1-00094's published +$538.99 or the main-grid flash_wick +$225.31
on dollars, but it is a much sturdier both-window / low-DD print than the
split-fragile VWAP all-in leader.

A close cousin with 9% of $280 (H4-01460) made +$79.61 OOS and +48.3% IS. The
requested "Keltner break gated by a washout" is the batch 4 quality find. The
requested "BB retest gated by SOL risk-on" did not work here.

## Do not repeat in later batches

Everything already listed for batches 1–3, **plus** this batch's 40 confluence
pairs, the SuperTrend 7/10/14 × 2/3 × ADX grid, this ADX-Donchian grid, this FVG
grid, this WVF grid, this Donchian-mid grid, and these five exits and five sizing
rules.

Still open: consecutive-loss cooldowns, ATR-sized stops, other gates that use a
*token* washout in front of unused entries, and any future dataset with wallet
flow or two venues per token. SOL-beta gates and ADX-Donchian breaks look like
dead ends on this universe unless a very different parameterization is tried.
Partial scale-outs still cannot be run in the all-in/all-out engine.

## Research inputs (batch 5)

Expanded the *token-washout gate* idea that held up in batches 3–4: an entry
fires only if the *same token* just printed a large single-hour dump. New
Keltner lengths (10/16/18/24h) and multipliers (1.8/2.2/2.8), VWAP windows
(10/14/15/18h), WVF periods (16/18/20/24) and SuperTrend (8/12/16h × 2.5/3.5)
were paired with unused dump thresholds (8/10/14/18/25%) and lookbacks
(10/16/20/30/48h). Fresh families came from classic systematic literature
adapted to hourly crypto: Elder Impulse (EMA slope + MACD-histogram slope),
Connors-style RSI dips above a slow SMA, MACD-histogram troughs / zero-line
crosses, and hourly Double-N (Double 7s: first bar of an N-bar closing low
while above a slow SMA).

From now on the *final shortlist is ranked by rolling walk-forward
robustness*, not by the single 60/40 OOS profit. For each batch's top 20
both-window names, plus H4-01464, H3-05689, H3-05569 and H2-02108, we ran
the same 26-fold design used on H1-00094 in batch 4 (original 60/40,
anchored 40/50/60/70/80%, calendar month-end, and three rolling schedules).
A name is written to `robust_leaderboard.csv` if, across **all 26 folds**:
full-period return > 0, median OOS return > 0, at least 50% of folds are
OOS-profitable, and at least 10 folds have one or more OOS trades. The
board itself is sorted by rolling fold-profitable %, then rolling median
OOS, then full-period return. A stricter "rolling-only" reading of the
same thresholds is also reported; it is not the official pass bar.

## Batch 5 families (10,000 new variants) + shortlist walk-forward

80 entries × 5 new exits × 5 new sizing rules × 5 new balances.

- **washout_gate** (48): Keltner / VWAP / WVF / SuperTrend entries gated by a
  token-level hourly dump, with the new lengths, thresholds and windows
  listed above. Example: SuperTrend 8h×3.5 only after a 14% dump in 20h;
  10h Keltner 2.8×ATR only after an 18% dump in 48h.
- **elder_impulse** (8): first green Elder bar, or a histogram up-turn while
  the EMA is rising, across EMA 8/13/21 and MACD 12/26/9, 8/21/5, 5/35/5.
- **connors_rsi** (8): RSI(2 or 3) crossing below 5/8/10/15 while price is
  above SMA 72/168/336/480.
- **macd_hist_turn** (8): histogram trough (any, below zero, or with EMA 13
  rising) or a zero-line cross, several MACD lengths.
- **double_n** (8): first hour of a 5/7/12/16/24-bar closing low above SMA
  48/72/168/240/336.

Exits (all new): tight 18%/−20%/60h scalp; +55%/−24% with a 16% trail after
+35% and a 180h cap; +80%/−27%/252h; no take-profit, −32% stop, 24% trail
after +55%, 432h cap; +120%/−19% with an 15% trail after +38% and a 336h
cap.

Sizing (all new pairs): 3.5%/28, 7.5%/13, 13%/8, 17%/5, 21%/4.
Balances: $45, $90, $160, $200, $300.

Walk-forward of the 101-name shortlist is *not* counted in the 10,000.

## Batch 5 results

10,000 variants in 39.3s on 4,297 hours × 139 tokens, 60/40 split at bar
2,578. 1,113 were OOS-profitable; **627 were profitable in both windows**
(412 of those with at least 8 OOS and 4 IS trades); 486 were OOS-only
flukes. Almost all of the both-window mass is in the washout-gate family
(622). Double-N produced 3 both-window names (best: H5-09969). Elder
Impulse and MACD-histogram turn produced one tiny both-window name each
(about $2.85 on a $45 account). Connors RSI had 8 OOS-profitable names
and **zero** both-window names.

On the original 60/40 split the top both-window name is **H5-01972**: 10h
Keltner (EMA+2.8×ATR) break only after an 18% dump in 48h, −32% stop, 24%
trail after +55%, 432h cap, 21% of $160 up to 4 open. OOS profit $489.27
(+305.8%, 25 trades, 19 IS trades, IS +22.6%). Walk-forward is more
modest: 52.94% of rolling folds profitable, rolling median OOS +6.16%,
full-period +118.1%. Several close sizing variants of the same entry sit
just behind it on the single split and tell the same story.

The *fresh* family that held up is Double-N. **H5-09969** (16-bar closing
low above the 240h SMA, same 432h trailing exit, 17% of $300 up to 5 open)
made $353.02 in the original OOS window (+117.7%, 43/43 trades, IS +21.2%)
and is the best *new* name on rolling walk-forward: **82.35% of rolling
folds profitable, rolling median OOS +18.61%, full-period +33.0%** on 76
trades. (Window returns always restart at the starting balance, so the
$353 OOS figure is not a live path that also traded the IS period; the
honest full-period path on $300 ends at $399.07.)

The batch 5 top 20 both-window names, ranked by rolling robustness:

- H5-09969 Double-N 16/240, 17% of $300: roll 82.35%, med +18.61%, full +33.0% — **pass**
- H5-04974 SuperTrend 8h×3.5 gated by a 14% dump in 20h, 21% of $300: roll 76.47%, med +34.92%, full +106.8% — **pass**
- H5-04973 same entry, 21% of $200: roll 70.59%, med +22.51%, full +276.5% — **pass**
- H5-04969 same entry, 17% of $300: roll 70.59%, med +15.79%, full +176.5% — **pass**
- H5-05099 SuperTrend 8h×3.5 gated by an 18% dump in 10h, 21% of $300: roll 70.59%, med +13.61%, full +174.8% — **pass**
- then several 58–53% rolling variants of the SuperTrend 18%/10h and 10h Keltner 18%/48h gates, including the single-split leader H5-01972 (roll 52.94%, med +6.16%, full +118.1%)
- 4 of the 20 pass the all-folds bar but miss the stricter rolling bar (rolling median slightly negative, rolling profitable share 47%)

### Named-leader walk-forward (current 139-token universe)

The same 26 folds as batch 4, rebuilt from saved params. The universe has
grown (118 tokens when H1-00094 was first published, 130 at the batch 4
retest, 139 now), so single-split numbers do not replay exactly; every
name below is judged on the *same* current data.

- **H2-02108 survives, and is one of the most robust names in the whole
  shortlist.** 10% / 1.5x climax confirmation, −22% stop, 20% trail after
  +40%, 360h, 6% of $100 up to 15 open. All-folds 88.46% OOS-profitable
  (median +35.28%), rolling **82.35% / median +12.72%**, full-period
  +159.2% on 159 trades, max DD 15.9%. Published OOS was $128.33; current
  60/40 replay is $121.51 / 78 trades. A close cousin, H2-02117 (same
  entry and exit, 15% of $25 up to 6 open), has an even higher rolling
  median (+42.47%) and full-period +449.9%.
- **H3-05569 survives.** 12h VWAP reclaim gated by a 10%/1.5x climax in
  the last 6h, +150%/−28%/192h, 18% of $250 up to 5 open. All-folds
  76.92% (median +9.78%), rolling **70.59% / median +9.44%**, full +43.96%
  on 56 trades.
- **H3-05689 survives**, same entry with a 12h climax window, 12% of $250
  up to 8 open. All-folds 65.38% (median +16.83%), rolling **64.71% /
  median +11.00%**, full +31.65% on 95 trades. Current-data full return is
  well below the published +122.6% — the idea holds, the magnitude does
  not fully transfer to the larger universe.
- **H4-01464 survives the official all-folds bar but is weak on rolling
  folds.** 14h Keltner 2× squeeze gated by a 12% dump in 24h, 11% of $250
  up to 9 open. All-folds 65.38% (median +18.89%), full +59.57% (matches
  the published full return; this was the one named name that replayed
  the 60/40 split exactly). Rolling is only **47.06% profitable, median
  −0.38%**. Treat it as a borderline survivor, not a rolling-robust
  leader.
- **H1-00094 fails again.** 12h VWAP all-in on $250: all-folds 34.62%
  (median −18.44%), rolling 35.29% (median −15.23%), full +106.5% on 15
  trades. The current 60/40 replay is $226.46 / 8 trades versus the
  published $538.99 / 14 — another reminder that this name's original
  OOS was split- and universe-specific.

Batch 1's all-in 12h/48h VWAP and 1-trade climax names mostly fail (17 of
the 26 official failures are batch 1). Thin climax configs that looked
great on one split die once the split moves.

### Robust leaderboard

75 of 101 shortlisted names pass the official all-folds rule and are
written to `robust_leaderboard.csv`, already sorted by rolling
robustness. 52 of those 75 also clear the same thresholds on *rolling
folds only*. The official board is dominated by diversified climax
reversal (batch 2), the new washout gates and Double-N (batch 5), and
the VWAP-after-climax ensembles (batch 3). Top of the board:

1. H3-09335 green-stack 3×2% greens, 7% of $280 — roll 88.24%, med +26.55%, full +16.91% (191 trades; most consistent hit rate, modest economic size)
2. H5-09969 Double-N 16/240, 17% of $300 — roll 82.35%, med +18.61%, full +33.02%
3. H2-02108 climax 10%/1.5x, 6% of $100 — roll 82.35%, med +12.72%, full +159.18%
4. H2-02117 same climax, 15% of $25 — roll 76.47%, med +42.47%, full +449.87%
5. H5-04974 SuperTrend 8h×3.5 + 14%/20h dump, 21% of $300 — roll 76.47%, med +34.92%, full +106.83%
6. H4-00204 12h VWAP + 12%/24h dump, 2% of $250 — roll 76.47%, med +4.76%, full +37.35% (367 trades; small per-trade edge, very diversified)
7. H1-06495 climax 15%/4x, all-in $280 — roll 70.59%, med +25.73%, full +124.88% (only 16 full-period trades; treat as thin)
8. H5-04973 SuperTrend 8h×3.5 + 14%/20h dump, 21% of $200 — roll 70.59%, med +22.51%, full +276.52%
9. H3-05693 VWAP after 12h climax, 18% of $100 — roll 70.59%, med +18.48%, full +55.13%
10. H5-04969 SuperTrend 8h×3.5 + 14%/20h dump, 17% of $300 — roll 70.59%, med +15.79%, full +176.48%

H3-05569 is #14 on this board. H3-05689 also passes and ranks further
down. H4-01464 is on the official board (all-folds pass) but would drop
off a rolling-only list. H1-00094 is not on the board.

## Do not repeat in later batches

Everything already listed for batches 1–4, **plus** this batch's 48
washout-gate pairs, the Elder Impulse / Connors RSI / MACD-histogram-turn
/ Double-N grids above, and these five exits, five sizing rules, and five
balances.

Elder Impulse, Connors RSI, and MACD-histogram turns look like dead ends
on this universe unless a very different parameterization is tried.
Still open: consecutive-loss cooldowns, ATR-sized stops, gating *other*
unused entries (Double-N, green-stack, BB retest) by a token washout,
and any future dataset with wallet flow or two venues per token.
Partial scale-outs still cannot be run in the all-in/all-out engine.

## Research inputs (batch 6)

Gated previously unused entries — Double-N, green-stack, and Bollinger
retest — by a token-level hourly dump, with new lookbacks (8/10/14/18/20-bar
lows, new SMAs, new stack lengths, new BB lengths) and dump thresholds
(9/11/13/16/22%) not used as pairs before. Fresh families: Williams %R
cross-up from oversold, CCI hook from oversold, Hull MA turn/cross, and
bullish pin bars. Also walk-forwarded the main-sweep moonshot top 20
(by OOS profit) plus the $10-balance 1000% and 10000% take-profit leaders,
using the same 26-fold design and pass rule as batch 5.

## Batch 6 families (10,000 new variants) + moonshot walk-forward

80 entries × 5 new exits × 5 new sizing rules × 5 new balances.

- **washout_gate** (48): Double-N (16), green-stack (16), Bollinger retest (16),
  each gated by a token dump.
- **williams_r** (8): %R 7/10/14/21/28 crossing up from −80/−85/−90.
- **cci_hook** (8): CCI 10/14/20/30 crossing up from −80/−100/−150/−200.
- **hull_flip** (8): HMA 9/16/25/49 turning up, or a fast HMA crossing a slow one.
- **pin_bar** (8): lower wick 50–70% of the range, close in the top of the bar,
  optional ATR-range filter.

Exits (all new): +25%/−16%/48h; +70%/−21% with a 14% trail after +30% and a
120h cap; +140%/−26%/288h; no take-profit, −30% stop, 20% trail after +60%,
384h cap; +200%/−18% with a 17% trail after +42% and a 216h cap.

Sizing: 4.5%/22, 6.5%/16, 9.5%/10, 15.5%/6, 28%/3.
Balances: $35, $55, $85, $175, $320.

Moonshot walk-forward is *not* counted in the 10,000.

## Batch 6 results

10,000 variants in 21.9s on 4,297 hours × 142 tokens. 1,950 OOS-profitable;
**688 both-windows** (457 solid); 1,262 OOS-only flukes. Almost all
both-window mass is still washout-gate (665). Williams %R had 19 both-window
names, Hull MA 3, pin bar 1, CCI hook 0.

The original-split both-window leader is **H6-01099**: 8-bar closing low
above the 120h SMA, only after a 16% dump in 16h, −30% stop, 20% trail after
+60%, 384h, 28% of $320 up to 3 open. OOS **$1,149.92** (+359.4%, 38 trades)
but IS is only +0.13% (51 trades). Walk-forward still passes (rolling 76.47%,
median +23.0%, full +359.9%) but it ranks 16th of the batch 6 top 20 by
rolling robustness. A close cousin on $175, **H6-01098**, is the more
credible version of the same entry: OOS $700.09, IS +37.3%, rolling
**94.12% / median +67.7%, full +556.7%**. That name is now #1 on
`robust_leaderboard.csv`.

All 20 of the batch 6 top both-window names are Double-N washout variants,
and **all 20 pass** both the official all-folds bar and the stricter rolling
bar. The first six all print a 94.12% rolling hit rate.

### Moonshot walk-forward (main sweep)

24 configs: the top 20 moonshot names by OOS profit (really four unique
rules at $195–$215 — first 24h or 72h, no mcap cap, $10k hour volume, no
momentum filter, 50% of balance up to 2 open, **+400% take-profit and no
stop**, with or without a 50% trail after +200%) plus the $10 1000% and
10000% take-profit leaders and the best both-window $10 1000% name.

The sweep was run on 108 tokens; current data has 142. The published $215
400% leaders ($3,359 OOS) **replay at −$129.74** on the current 60/40 split.
Across folds they fail clearly: rolling 17–24% profitable, median OOS about
−34% to −38%. Full-period returns stay large (+189% to +1,159%) because a
few huge winners dominate a no-stop rule — that is not robustness.

**$10 / 1000% TP:** the raw leader (MS-4252024, first 24h, $1k volume, 50%
of $10, no stop, 720h) is an OOS-only fluke (IS −72.6%). Rolling 5.88%,
median −36%, full −29.6%. The best both-window $10 1000% name
(MS-3635077, first 6h, $10k volume, 34% of $10) also fails: rolling 23.53%,
median −32.1%, full +20.9%.

**$10 / 10000% TP:** the leader and the best both-window name
(MS-4715308 / MS-3892780) are the same exit — +10,000% TP, no stop, 50%
trail after +200%, 240h, 20% of $10 up to 5 open — with a 72h vs 24h
launch window and a $200k mcap cap. They **replay the published 60/40
exactly** ($33.46 OOS). They pass the official all-folds bar (61.54% of
folds profitable, median +7.5%, full +2,360% / +2,659%) but **fail
rolling walk-forward** (41.18% profitable, median −3.3%). Mean OOS is
far above the median; a few giant winners are doing the work.

**No moonshot config survives rolling walk-forward.** Two $10 10000%
names sit on the official robust board only because the all-folds bar is
more lenient.

## Do not repeat in later batches

Everything already listed for batches 1–5, **plus** this batch's 48
washout-gate pairs, the Williams %R / CCI hook / Hull flip / pin-bar
grids, and these five exits, five sizing rules, and five balances.

CCI hook looks like a dead end here. Williams %R, Hull, and pin bars were
weak. Still open: consecutive-loss cooldowns, ATR-sized stops, wallet flow
or two venues per token, and gating other unused entries by a token
washout. Partial scale-outs still cannot be run in the all-in/all-out
engine. Moonshot +400% / +1000% / +10000% take-profits with no stop should
not be retested on this universe without a very different entry or a hard
stop; they do not hold up under rolling walk-forward.

## Research inputs (batch 7)

The search is heavily mined, so this batch reserved a frozen holdout. No new
hourly bars arrived after batch 6 (still 4,297 hours, last bar 2026-11-13
03:00 PM CST). The holdout is therefore the most recent 14 days already in
the file: 336 bars from 2026-10-30 05:00 PM CDT to 2026-11-13 03:00 PM CST
(~19.5% of the old 60/40 OOS). Those bars *were* inside batches 1–6 OOS and
walk-forward, so they are not unseen for the existing robust board. Batch 7
itself was ranked only on data before the holdout (IS 0–2578, purged OOS
2578–3961). Walk-forward folds also stop at the holdout start.

Entries stay close to the H6 Double-N washout leaders (8/120 + 16% dump in
16h; 8/288 + 16% dump in 24h; 10/288 + 16% dump in 36h): nearby n (7/9/11),
SMAs (108/132/144/264/312/336), dump thresholds (14/15/17/18%) and windows
(14/18/20/22/28/30/32/40). No batch-6 pairs were repeated.

## Batch 7 families (10,000 new variants) + holdout check

80 Double-N washout entries × 5 new exits × 5 new sizing rules × 5 new
balances. Exits sit next to the H6 trail/TP winners (29%/33%/31% stops with
58–70% trail triggers, or +130–160% TPs). Sizing: 10.5%/9, 18.5%/5, 24%/4,
29%/3, 33%/3. Balances: $65, $110, $190, $240, $350.

## Batch 7 results

10,000 variants in 19.1s on the same 4,297 × 142 data. Ranked on purged
windows only: 7,748 purged-OOS profitable, **4,982 both-windows** (all solid),
2,766 purged-OOS flukes. 9,775 were profitable on the holdout slice; 4,925
were both-windows *and* holdout-profitable. That high hit rate is expected
in a tight neighborhood of a family that already worked on most of this tape.

Purged both-window leader: **H7-02419**, 8-bar low above the 132h SMA gated
by a 17% dump in 14h, −33% stop, 22% trail after +65%, 408h, 29% of $350
up to 3 open. Purged OOS **$1,352.24** (+386.4%, 22 trades, IS +71.2%). On
the unseen-to-batch-7 holdout it made **$551.61** (+157.6%, 14 trades, DD
15.2%). All 20 of the batch 7 top both-window names passed pre-holdout
walk-forward and all 20 were profitable on the holdout. They are mostly
8-bar / 108–144h SMA variants with 15–17% dumps over 14–16h.

### Holdout credibility check (main result)

Robust top 10, last 14 days (windows restart at the starting balance):

| name | family | holdout ret | profit | trades | dd | purged OOS ret |
|---|---|---:|---:|---:|---:|---:|
| H6-01098 | DN 8/120 + 16%/16h, 28% of $175 | +28.3% | $49.48 | 21 | 29.3% | +153.4% |
| H6-01094 | same entry, 15.5% of $320 | +24.7% | $79.16 | 20 | 12.3% | +121.2% |
| H6-01718 | DN 8/288 + 16%/24h, 15.5% of $175 | +37.7% | $65.90 | 35 | 15.0% | +111.6% |
| H6-01689 | same entry, +140% TP, 9.5% of $320 | +52.5% | $168.04 | 28 | 18.2% | +87.2% |
| H6-01064 | DN 8/120, +140% TP, 9.5% of $320 | +17.2% | $55.01 | 22 | 20.6% | +65.1% |
| H6-01714 | DN 8/288, 9.5% of $320 | +22.6% | $72.17 | 35 | 9.4% | +74.9% |
| H6-01097 | DN 8/120, 28% of $85 | +34.8% | $29.60 | 25 | 28.5% | +140.7% |
| H3-09335 | green-stack 3×2%, 7% of $280 | +3.1% | $8.60 | 23 | 7.8% | +27.3% |
| H6-00468 | DN 10/288 + 16%/36h, 15.5% of $175 | +74.5% | $130.45 | 33 | 10.7% | +85.4% |
| H6-00464 | same entry, 9.5% of $320 | +42.6% | $136.43 | 33 | 6.6% | +57.2% |

**10/10 profitable. 9/9 H6 Double-N names clearly so.** The only non-DN name,
H3-09335, barely cleared (+3.1%). Best holdout return in the top 10 is
H6-00468 (+74.5%); best dollars are H6-01689 ($168 on $320). The current
robust #1, H6-01098, made a calmer +28.3% / $49.

This is encouraging but not a clean out-of-sample test for those ten names:
the same two weeks were already inside their original OOS and walk-forward.
It *is* a clean holdout for batch 7, which never used those bars to rank,
and the nearby 8-bar / ~120h variants also made money there. The last 14
days were a good tape for this family, which will flatter both the old
leaders and the new neighborhood grid. Treat it as a consistency check,
not a green light to size up live.

20 batch-7 names (all of the top 20) were added to `robust_leaderboard.csv`
after passing pre-holdout walk-forward *and* the holdout. The board still
opens with the H6 Double-N leaders; pre-holdout walk-forward percentages
are not strictly comparable to the earlier 26-fold numbers.

## Batch 8 — 10,000 fresh families, frozen holdout

Shifted entirely off the mined Double-N neighborhood: 8 new families × 10
entries (100% fresh, well above the 70% floor). TRIX, Coppock, KST, Awesome
Oscillator (zero-cross and saucer), Ichimoku cloud (no-lookahead Senkou),
Stochastic RSI (stoch of RSI, not batch-2 stoch_uptrend), Vortex, and TSI.
Ichimoku cloud at bar t uses Tenkan/Kijun/span values from t−kijun.

New exits: +22%/−17%/42h; +65%/−23% trail 15% after +32%/156h; +125%/−29%/276h;
no TP, −27% trail 19% after +48%/396h; +180%/−20% trail 16% after +44%/228h.
Sizing: 5.5% of balance / 18 open, 11.5%/8, 16.5%/6, 27%/3, 36%/2.
Balances: $48, $72, $125, $215, $340.

Ranking used the same frozen holdout: IS [0, 2578), purged OOS [2578, 3961),
walk-forward end_bar=3961. A GeckoTerminal probe (1 request, no 429s) found
no complete hourly bar newer than 2026-11-13 03:00 PM CST, already the last
bar in the file. Nothing was appended; `/workspace/solstrats` was not
modified. There is no truly unseen post-4297 holdout this batch.

Of 10,000 variants: 560 purged-OOS profitable, 101 both-windows, 459
OOS-only flukes, 12 both-windows *and* frozen-holdout profitable. Family
both-window counts: awesome_osc 26, tsi_cross 19, coppock_turn 14,
stoch_rsi 14, vortex_cross 12, ichimoku_cloud 8, trix_cross 7, kst_cross 1.

Purged both-window dollar leader H8-01724 (Coppock ROC 11+14 WMA 10 zero
cross) made $605.65 on purged OOS but failed walk-forward (40% fold
profitable, full-period −49.8%) and lost 9.6% on the holdout. Same pattern
for several large Coppock names: big single-split OOS, weak folds.

6 of the top 20 both-window names passed official pre-holdout walk-forward.
None of those 6 were profitable on the frozen 14-day holdout, so **no
batch-8 name was added** to `robust_leaderboard.csv` (still 117).

Walk-forward passers (all lost money on the frozen holdout):

- H8-08461 Vortex 32, −27%/trail 19% after +48%/396h, 16.5% of $72:
  roll 80%, median OOS +15.7%, full +79.9%; hold −14.3% / −$10.31
- H8-04206 AO 7/28 saucer, same exit, 11.5% of $72:
  roll 73.3%, med +14.7%, full +24.1%; hold −8.4%
- H8-04697 AO 6/40 saucer, +125%/−29%/276h, 36% of $125:
  roll 73.3%, med +11.2%, full +48.2%; hold −34.9%
- H8-06964 StochRSI(7,7) from below 0.2, −27%/trail 19% after +48%/396h,
  16.5% of $340: roll 66.7%, med +31.8%, full +112.2%; hold −10.4% / −$35.50
- H8-04207 / H8-04208 same AO 7/28 saucer & exit as 04206, 11.5% of $125 /
  $215: roll 66.7%; hold −16.4% and −16.8%

Two top-20 Coppock names (H8-01872 / H8-01871, ROC 24+48 WMA 14 zero-cross,
+180% trail exit, 36% of $125/$72) were slightly profitable on the holdout
(+1.6% / +2.2%) but failed walk-forward.

Current robust-board top 10 on the same frozen 14 days (2026-10-30 05:00 PM
CDT → 2026-11-13 03:00 PM CST): still 10/10 profitable. #1–#6 are the H6
Double-N leaders (unchanged vs batch 7). #7–#10 are now batch-7 Double-N
neighbors (H7-02417, H7-02423, H7-00668, H7-06668) that were required to
be profitable on this window to join the board, so those four holdout
figures (+124% to +177%) are not an independent check. No new hours exist
to re-rank anyone on a truly unseen sliver.

These fresh families did not displace Double-N on a 14-day tape that was
already known to favor that neighborhood. Do not overclaim a live edge.

## Do not repeat in later batches

Everything already listed for batches 1–7, **plus** this batch's eight
families (trix_cross, coppock_turn, kst_cross, awesome_osc, ichimoku_cloud,
stoch_rsi, vortex_cross, tsi_cross) and these five exits, five sizing
rules, and five balances. Keep the holdout frozen at bar 3961 unless
genuinely new hours are fetched. Still open: consecutive-loss cooldowns,
ATR-sized stops, wallet flow or two venues, and a true post-batch-6 data
refresh.
