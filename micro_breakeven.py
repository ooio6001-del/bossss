"""Micro buy-size study (same max open positions as each strategy) for the headline (walk-forward robust) and top strategies: fixed $ per trade with the real fee
model (DEX fee + slippage per side, impact, base+priority SOL fee per swap, unrecovered token-account rent).
Break-even buy size = smallest fixed buy size whose simulated total P&L is positive (fills capped at 5% of bar volume,
so tiny-volume moonshot bars can still be skipped). The naive figure (fixed round-trip cost / mean net % per trade) is also given. -> results/micro_breakeven.{json,csv}"""
from __future__ import annotations
import csv, json, sys
import numpy as np
from strategies import Registry
from compute_windows import strategy_rows
from solstrats import engine as E, costs as K
from run_backtest import prep

SIZES = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0, 2.0, 5.0, 10.0, 25.0]

def main(R="results"):
    D = prep(); reg = Registry(D, R); T = int(D["T"])
    sp_avg = float(np.mean(D["solpx"])); fixed_rt = (2 * K.FIXED_SOL + K.RENT_SOL * K.RENT_UNRECOVERED) * sp_avg
    rows = strategy_rows(R); pick, seen = [], set()
    for tag, n in (("robust", 10), ("top100", 6), ("top100_profit", 4)):
        k = 0
        for r in rows:
            if r["_tag"] != tag or k >= n: continue
            s = reg.spec(r); key = (s or {}).get("sig_key"), tuple((s or {}).get("e", []))
            if s and key not in seen:
                seen.add(key); s["tag"] = tag; s["entry"] = r.get("entry", ""); s["exit"] = r.get("exit", ""); pick.append(s); k += 1
    out = []
    for s in pick:
        tok, ti, to, pin, pout, rs = reg.trades(s, 0)
        if tok.size == 0: continue
        ve = D["V"][np.maximum(ti - 1, 0), tok]
        pc = K.PCT_COST[D["venue"][tok]]
        net = (pout / pin) * (1 - pc) ** 2 - 1
        mean = float(net.mean()); be_naive = fixed_rt / mean if mean > 0 else None
        def pnl(sz):
            m = E.simulate(tok, ti, to, pin, pout, D["venue"], ve, D["solpx"], 0, T, 1000.0, -sz, int(s["maxpos"]), K.PCT_COST, K.FIXED_SOL,
                           K.RENT_SOL, K.RENT_UNRECOVERED, K.IMPACT_K, K.FILL_CAP, np.zeros(0))
            return float(m[0] - 1000), int(m[4])
        # simulated break-even: smallest fixed buy size with total P&L > 0 (bisection on log size, $0.01..$250)
        be = None
        grid = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 25, 50, 100, 250]
        prev = None
        for g in grid:
            if pnl(g)[0] > 0:
                if prev is None: be = g
                else:
                    lo_, hi_ = np.log(prev), np.log(g)
                    for _ in range(25):
                        mid = (lo_ + hi_) / 2
                        if pnl(float(np.exp(mid)))[0] > 0: hi_ = mid
                        else: lo_ = mid
                    be = float(np.exp(hi_))
                break
            prev = g
        sims = []
        for sz in SIZES:
            m = E.simulate(tok, ti, to, pin, pout, D["venue"], ve, D["solpx"], 0, T, 1000.0, -sz, int(s["maxpos"]), K.PCT_COST, K.FIXED_SOL,
                           K.RENT_SOL, K.RENT_UNRECOVERED, K.IMPACT_K, K.FILL_CAP, np.zeros(0))
            n = max(1, int(m[4]))
            sims.append({"size": sz, "trades": int(m[4]), "pnl": round(float(m[0] - 1000), 4), "pnl_per_trade": round(float((m[0] - 1000) / n), 5)})
        out.append({"id": s["id"], "tag": s["tag"], "family": s["family"], "entry": s["entry"], "trades": int(tok.size),
                    "avg_net_pct": round(mean, 4), "fixed_rt_usd": round(fixed_rt, 4),
                    "breakeven_usd": None if be is None else round(be, 3),
                    "breakeven_naive_usd": None if be_naive is None else round(be_naive, 3), "sims": sims})
    json.dump({"fixed_rt_usd": fixed_rt, "sol_avg": sp_avg, "sizes": SIZES, "rows": out}, open(f"{R}/micro_breakeven.json", "w"))
    with open(f"{R}/micro_breakeven.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "board", "family", "entry", "trades", "avg_net_pct_per_trade", "fixed_round_trip_usd", "breakeven_buy_usd_simulated", "breakeven_naive_usd"] + [f"pnl_per_trade_at_${s}" for s in SIZES])
        for r in out:
            w.writerow([r["id"], r["tag"], r["family"], r["entry"], r["trades"], r["avg_net_pct"], round(fixed_rt, 4), r["breakeven_usd"], r["breakeven_naive_usd"]] + [x["pnl_per_trade"] for x in r["sims"]])
    for r in out: print(r["id"], r["tag"], r["trades"], r["avg_net_pct"], r["breakeven_usd"], [x["pnl_per_trade"] for x in r["sims"][:6]])
    print("fixed round trip $", round(fixed_rt, 4))

if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["results"]))
