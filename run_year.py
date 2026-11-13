"""12-month 'Best of the Year' sweep on DAILY CoinGecko closes for established coins (GeckoTerminal public hourly
history stops at 180 days). Same engine; parameters are in DAYS here; stops/targets are checked on daily closes only
(no intraday highs/lows available), so results are coarser than the hourly sweep."""
import glob, itertools, json, time
import numpy as np
from solstrats import engine as E, costs as K, grid as G
from solstrats.describe import exit_text, sizing_text

def load():
    fs = sorted(glob.glob("data/daily/*.json")); recs = []
    for f in fs:
        d = json.load(open(f))
        if len(d["prices"]) >= 200: recs.append(d)
    t_end = max(r["prices"][-1][0] for r in recs) // 86400000; t0 = t_end - 364
    T = 365; N = len(recs)
    C = np.full((T, N), np.nan); V = np.zeros((T, N))
    for j, r in enumerate(recs):
        for (ts, p), (_, v) in zip(r["prices"], r["vols"]):
            i = int(ts // 86400000 - t0)
            if 0 <= i < T: C[i, j] = p; V[i, j] = v
        for i in range(1, T):
            if np.isnan(C[i, j]) and not np.isnan(C[i - 1, j]): C[i, j] = C[i - 1, j]
    # remove one-day bad ticks (price collapses/explodes >4x and fully reverts the next day): data errors, not tradable
    fixed = 0
    for j in range(N):
        for i in range(1, T - 1):
            a, b, c = C[i - 1, j], C[i, j], C[i + 1, j]
            if a > 0 and b > 0 and c > 0 and ((b / a < 0.25 and c / b > 3) or (b / a > 4 and c / b < 0.33)):
                C[i, j] = a; fixed += 1
    # exclude coins whose feed shows a V-shaped >80% collapse fully recovered within 10 days (feed errors / token
    # migrations such as re-denominations); these produce fake 10-30x "dip-buy" trades
    bad = []
    for j in range(N):
        c = C[:, j]
        for i in range(T - 10):
            w = c[i:i + 11]
            if np.isnan(w).any() or w[0] <= 0: continue
            if w.min() < 0.2 * w[0] and w[-1] > 0.6 * w[0]:
                bad.append(recs[j]["sym"]); break
    keep = [j for j in range(N) if recs[j]["sym"] not in bad]
    C, V = C[:, keep], V[:, keep]; recs = [recs[j] for j in keep]
    O = C.copy(); H = C.copy(); L = C.copy()   # closes only: entries fill at the next daily close, stops/targets fill at the close that breaches them
    load.fixed = fixed; load.excluded = bad
    return recs, O, H, L, C, V, t0

def main():
    t_start = time.time()
    recs, O, H, L, C, V, t0 = load(); T, N = C.shape
    syms = [r["sym"] for r in recs]; sol = syms.index("SOL") if "SOL" in syms else -1
    valid = ~np.isnan(C); first = np.where(valid.any(0), valid.argmax(0), -1).astype(np.int64)
    last = np.where(valid.any(0), T - 1 - valid[::-1].argmax(0), -1).astype(np.int64)
    grp = np.zeros(N, np.int64); grp[sol] = 2
    z = np.zeros(N); zi = np.zeros(N, np.int64); solpx = C[:, sol] if sol >= 0 else np.full(T, 150.0)
    ent = ([("breakout", (L_, th, 0, 0, 0, 0)) for L_, th in itertools.product((5, 10, 20, 50), (0.0, 0.05))] +
           [("ma_cross", (f, s, 0, 0, 0, 0)) for f, s in ((5, 20), (10, 50), (20, 100), (3, 10))] +
           [("rsi_revert", (p, t, tr, 0, 0, 0)) for p, t, tr in itertools.product((6, 14), (20, 30), (0,))] +
           [("vol_spike", (vm, r, 0, 0, 0, 0)) for vm, r in itertools.product((2, 3, 5), (0.05, 0.15))] +
           [("dip_buy", (L_, d, b, 0, 0, 0)) for L_, d, b in itertools.product((10, 30), (0.3, 0.5), (0, 1))])
    exits = [(tp, sl, a, w, tm, E.NONE, 1.0) for tp in (0.25, 0.5, 1.0, 2.0, E.NONE) for sl in (0.1, 0.2, 0.35)
             for a, w in ((E.NONE, 0), (0.3, 0.2)) for tm in (10, 30, 90)]
    combos, parts = [], []
    for fam, p in ent:
        sig = E.signals(E.F[fam], np.asarray(p, float), O, H, L, C, V, first, grp, zi, np.full(N, 1e9), z, z, np.full(N, np.nan),
                        np.zeros(T, np.int64), zi, 0, T - 1)
        for e in exits:
            parts.append(E.gen_trades(sig, np.asarray(e, float), O, H, L, C, last, np.full(N, 1e9), False)); combos.append((fam, p, e))
    lens = np.array([len(t[0]) for t in parts]); offs = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    tok, ti, to, pin, pout, rs = [np.concatenate([t[k] for t in parts]) for k in range(6)]
    tok = tok.astype(np.int64); ti = ti.astype(np.int64); to = to.astype(np.int64)
    ve = V[np.maximum(ti - 1, 0), tok]
    nb, ns, nc = len(K.BALANCES), len(G.SIZING), len(combos)
    cc = np.repeat(np.arange(nc), ns * nb).astype(np.int64); cs = np.tile(np.repeat(np.arange(ns), nb), nc); cbal = np.tile(K.BALANCES, nc * ns)
    fr = np.array([s[0] for s in G.SIZING])[cs]; mp = np.array([s[1] for s in G.SIZING], np.int64)[cs]
    # sim uses bars-per-day=24 for the daily Sharpe bucket; on daily bars pass a 24x-expanded index by scaling time
    res = E.simulate_many(tok * 1, ti * 24, to * 24, pin, pout, zi, ve, np.repeat(solpx, 24), offs, cc, cbal, fr, mp, 0, T * 24,
                          K.PCT_COST, K.FIXED_SOL, K.RENT_SOL, K.RENT_UNRECOVERED, K.IMPACT_K, K.FILL_CAP)
    sp = int(T * 0.6) * 24
    res_is = E.simulate_many(tok, ti * 24, to * 24, pin, pout, zi, ve, np.repeat(solpx, 24), offs, cc, cbal, fr, mp, 0, sp,
                             K.PCT_COST, K.FIXED_SOL, K.RENT_SOL, K.RENT_UNRECOVERED, K.IMPACT_K, K.FILL_CAP)
    res_oos = E.simulate_many(tok, ti * 24, to * 24, pin, pout, zi, ve, np.repeat(solpx, 24), offs, cc, cbal, fr, mp, sp, T * 24,
                              K.PCT_COST, K.FIXED_SOL, K.RENT_SOL, K.RENT_UNRECOVERED, K.IMPACT_K, K.FILL_CAP)
    ok = res[:, 4] >= 10
    idx = np.where(ok)[0]; idx = idx[np.argsort(-res[idx, 1])]
    rows, seen = [], set()
    for i in idx:
        k = (int(cc[i]), int(cs[i]))
        if k in seen: continue
        seen.add(k); fam, p, e = combos[cc[i]]
        txt = {"breakout": f"Buy when price closes above its {p[0]}-day high" + (f" by {p[1]*100:g}%" if p[1] else ""),
               "ma_cross": f"Buy when the {p[0]}-day average crosses above the {p[1]}-day average",
               "rsi_revert": f"Buy when {p[0]}-day RSI drops below {p[1]}", "vol_spike": f"Buy a +{p[1]*100:g}% day on {p[0]}x average volume",
               "dip_buy": f"Buy after a {p[1]*100:g}% drop from the {p[0]}-day high" + (" once a day closes green" if p[2] else "")}[fam]
        rows.append({"family": fam, "entry": txt, "exit": exit_text(e).replace("h", " days", 1) if False else exit_text((e[0], e[1], e[2], e[3], e[4], e[5], e[6])).replace(f"after {int(e[4])}h", f"after {int(e[4])} days"),
                     "sizing": sizing_text(*G.SIZING[cs[i]]), "balance": float(cbal[i]), "end": round(float(res[i, 0]), 2),
                     "ret": round(float(res[i, 1]), 4), "win": round(float(res[i, 13]), 4), "trades": int(res[i, 4]), "dd": round(float(res[i, 2]), 4),
                     "sharpe": round(float(res[i, 3]), 3), "is_ret": round(float(res_is[i, 1]), 4), "oos_ret": round(float(res_oos[i, 1]), 4),
                     "oos_trades": int(res_oos[i, 4])})
        if len(rows) >= 100: break
    bh = [(syms[j], float(C[last[j], j] / C[first[j], j] - 1)) for j in range(N) if first[j] >= 0]
    out = {"configs": int(cc.size), "strategies": nc, "coins": N - (1 if sol >= 0 else 0), "start": time.strftime("%Y-%m-%d", time.gmtime(t0 * 86400)),
           "end": time.strftime("%Y-%m-%d", time.gmtime((t0 + 364) * 86400)), "top": rows,
           "basket_bh": round(float(np.mean([r for s, r in bh if s != "SOL"])), 4), "sol_bh": round(dict(bh).get("SOL", float("nan")), 4),
           "elapsed_s": round(time.time() - t_start), "split": time.strftime("%Y-%m-%d", time.gmtime((t0 + int(T * 0.6)) * 86400)),
           "oos_positive_share_top100": round(float(np.mean([r["oos_ret"] > 0 for r in rows])), 3) if rows else None, "bad_ticks_removed": load.fixed, "excluded_feed_errors": load.excluded}
    json.dump(out, open("results/year.json", "w"))
    print("year sweep", out["configs"], "configs", out["coins"], "coins", out["start"], "->", out["end"], "best", rows[0]["ret"] if rows else None)

if __name__ == "__main__":
    main()
