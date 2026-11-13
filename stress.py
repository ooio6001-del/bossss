"""Stress tests for the walk-forward robust strategies (full hourly span, each strategy's own balance and sizing, same engine):
  costs x1.5 / x2 / x3 (DEX fee+slippage, SOL fees, impact), entry one bar late, drop whichever single trade hurts most, drop every trade on
  whichever token hurts most, and a random 80% trade subsample (500 draws: 5th percentile and share profitable).
A strategy "survives" when it stays profitable at 2x costs, one bar late, and without its best token. -> results/stress.{json,csv}"""
from __future__ import annotations
import csv, json, os, sys
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".numba_cache"))
sys.dont_write_bytecode = True
import numpy as np
from strategies import Registry
from compute_windows import strategy_rows
from solstrats import engine as E, costs as K
from run_backtest import prep

N_ROBUST = int(os.environ.get("STRESS_N", "200")); DRAWS = 500

def sim(D, tr, bal, frac, mp, cm=1.0):
    tok, ti, to, pin, pout = tr[:5]
    if tok.size == 0: return 0.0, 0
    ve = D["V"][np.maximum(ti - 1, 0), tok]
    m = E.simulate(tok, ti, to, pin, pout, D["venue"], ve, D["solpx"], 0, int(D["T"]), bal, frac, int(mp), K.PCT_COST * cm,
                   K.FIXED_SOL * cm, K.RENT_SOL, K.RENT_UNRECOVERED, K.IMPACT_K * cm, K.FILL_CAP, np.zeros(0))
    return float(m[1]), int(m[4])

def sub(tr, keep):
    return tuple(a[keep] for a in tr)

def stress_one(D, reg, s, rng):
    tr = reg.trades(s, 0); bal, fr, mp = s["bal"], s["frac"], s["maxpos"]
    base, n = sim(D, tr, bal, fr, mp)
    r = {"id": s["id"], "family": s["family"], "balance": bal, "trades": n, "base": base}
    for cm in (1.5, 2.0, 3.0): r[f"cost_x{cm:g}"] = sim(D, tr, bal, fr, mp, cm)[0]
    sig = reg.signal(s); late = np.zeros_like(sig); late[1:] = sig[:-1]
    trl = E.gen_trades(late, np.asarray(s["e"], float), D["O"], D["H"], D["L"], D["C"], D["last"], D["supply"], bool(s["one_shot"]))
    r["late_1bar"] = sim(D, trl, bal, fr, mp)[0]
    tok, pin, pout = tr[0], tr[3], tr[4]
    if tok.size:
        # worst case over every single trade / every single token removed (the engine skips some signals, so test them all)
        allk = np.ones(tok.size, bool); bests = []
        for i in range(tok.size):
            allk[i] = False; bests.append(sim(D, sub(tr, allk), bal, fr, mp)[0]); allk[i] = True
        r["drop_best_trade"] = float(min(bests))
        tk = {j: sim(D, sub(tr, tok != j), bal, fr, mp)[0] for j in np.unique(tok).tolist()}
        bt = min(tk, key=tk.get); r["best_token"] = str(D["metas"][bt].get("sym", bt)); r["drop_best_token"] = float(tk[bt])
        subs = np.array([sim(D, sub(tr, rng.random(tok.size) < 0.8), bal, fr, mp)[0] for _ in range(DRAWS)])
        r["sub80_p05"] = float(np.percentile(subs, 5)); r["sub80_median"] = float(np.median(subs)); r["sub80_profitable"] = float((subs > 0).mean())
    r["survives"] = bool(r["base"] > 0 and r["cost_x2"] > 0 and r["late_1bar"] > 0 and r.get("drop_best_token", -1) > 0)
    return r

def main(R="results"):
    D = prep(); reg = Registry(D, R); rng = np.random.default_rng(7)
    rows = [r for r in strategy_rows(R) if r["_tag"] == "robust"][:N_ROBUST]
    out, seen = [], set()
    for row in rows:
        s = reg.spec(row)
        if not s: continue
        k = (s["sig_key"], tuple(s["e"]), s["frac"], s["maxpos"], s["bal"])
        if k in seen: continue
        seen.add(k); out.append(stress_one(D, reg, s, rng))
    keys = ["id", "family", "balance", "trades", "base", "cost_x1.5", "cost_x2", "cost_x3", "late_1bar", "drop_best_trade", "best_token",
            "drop_best_token", "sub80_p05", "sub80_median", "sub80_profitable", "survives"]
    with open(f"{R}/stress.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore"); w.writeheader(); [w.writerow(r) for r in out]
    summ = {"n": len(out), "survivors": sum(r["survives"] for r in out),
            "profitable_at": {k: sum(r.get(k, -1) > 0 for r in out) for k in ("base", "cost_x1.5", "cost_x2", "cost_x3", "late_1bar", "drop_best_trade", "drop_best_token")}}
    json.dump({"summary": summ, "rows": out}, open(f"{R}/stress.json", "w"))
    for r in out: print(r["id"], r["balance"], r["trades"], *(f"{k}={r.get(k, 0):+.2f}" for k in ("base", "cost_x2", "cost_x3", "late_1bar", "drop_best_token", "sub80_p05")), r["survives"])
    print(summ)

if __name__ == "__main__":
    main()
