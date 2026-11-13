"""Precompute exact date-range results for every preset and export the client-side payload for dashboard.html.
Usage: python3 compute_windows.py [results_dir]   -> results/windows.json, results/window_payload.json"""
from __future__ import annotations
import base64, gzip, json, sys, time
import numpy as np
from strategies import Registry, read_csv, HUNT
from solstrats import window as W, costs as K
from run_backtest import prep

PRESETS = ["1h", "4h", "12h", "24h", "48h", "7d", "30d", "90d", "ytd", "12m"]

def strategy_rows(R, n_hunt=200):
    rows, seen = [], set()
    def add(r, tag):
        k = (r["id"], float(r["balance"]), r.get("sizing", ""))
        if k in seen: return
        seen.add(k); r = dict(r); r["_tag"] = tag; rows.append(r)
    for r in read_csv(f"{HUNT}/robust_leaderboard.csv"):
        if str(r.get("passed", "true")).lower() == "true": add(r, "robust")
    for r in read_csv(f"{R}/top100.csv"): add(r, "top100")
    for r in read_csv(f"{R}/top100_by_profit.csv"): add(r, "top100_profit")
    for r in read_csv(f"{HUNT}/leaderboard.csv")[:n_hunt]: add(r, "hunter")
    return rows

def b64(a, dtype):
    return base64.b64encode(gzip.compress(np.ascontiguousarray(a, dtype=dtype).tobytes(), 6)).decode()

def main():
    R = sys.argv[1] if len(sys.argv) > 1 else "results"
    t0 = time.time(); D = prep(); reg = Registry(D, R)
    rows = strategy_rows(R); specs = []
    for r in rows:
        s = reg.spec(r)
        if s: s["tag"] = r["_tag"]; s["entry"] = r.get("entry", ""); s["exit"] = r.get("exit", ""); specs.append(s)
    print("strategies", len(specs), "of", len(rows), flush=True)
    out = {"generated": time.strftime("%Y-%m-%d %H:%M %Z"), "presets": {}, "n_strategies": len(specs)}
    for p in PRESETS:
        b = W.resolve(D, preset=p); res = []
        for s in specs:
            tr = W.replay(reg.candidates(s), b["t_lo"], s["one_shot"])
            m = W.pack(W.run(D, tr, b["t_lo"], b["t_hi"], s["bal"], s["frac"], s["maxpos"]), s["bal"])
            res.append({"i": len(res), **m})
        out["presets"][p] = {"bounds": b, "rows": res}
        print(p, b["start"], "->", b["end"], "capped" if b["capped_to_data"] else "", f"{time.time()-t0:.0f}s", flush=True)
    out["strategies"] = [{"id": s["id"], "family": s["family"], "entry": s["entry"], "exit": s["exit"], "sizing": s["sizing"],
                          "balance": s["bal"], "tag": s["tag"], "frac": s["frac"], "maxpos": s["maxpos"]} for s in specs]
    json.dump(out, open(f"{R}/windows.json", "w"))
    # client payload: gen_trades_all candidates per (signal, exit, one_shot) group + int16 log-close matrix
    gid, groups = {}, []
    for s in specs:
        k = (s["sig_key"], tuple(map(float, s["e"])), bool(s["one_shot"]))
        if k not in gid:
            gid[k] = len(groups); groups.append(reg.candidates(s))
        s["g"] = gid[k]
    tok, ti, to, gr, pin, ve, eo, off = [], [], [], [], [], [], [], [0]
    for t in groups:
        tok.append(t[0]); ti.append(t[1]); to.append(t[2]); gr.append(t[4] / t[3]); pin.append(t[3])
        ve.append(D["V"][np.maximum(t[1] - 1, 0), t[0]]); eo.append(W.end_open(D, t)); off.append(off[-1] + t[0].size)
    pins = np.concatenate(pin)
    C = D["C"].copy()
    for j in range(C.shape[1]):  # forward-fill each token's closes
        col = C[:, j]; m = np.isfinite(col) & (col > 0)
        if m.any():
            idx = np.where(m, np.arange(col.size), 0); np.maximum.accumulate(idx, out=idx)
            col[:] = col[idx]; col[~np.cumsum(m).astype(bool)] = np.nan
    lc = np.log(np.where(np.isfinite(C) & (C > 0), C, np.nan))
    base = np.nanmin(np.where(np.isfinite(lc), lc, np.inf), axis=0); base[~np.isfinite(base)] = 0
    q = np.round((lc - base) * 1000); q = np.where(np.isfinite(q), np.clip(q, 0, 32766), 32767).astype(np.int16)
    payload = {"T": int(D["T"]), "N": int(C.shape[1]), "t0": float(D["times"][0]), "venue": [int(x) for x in D["venue"]],
               "solpx": [round(float(x), 3) for x in D["solpx"]], "pct_cost": list(map(float, K.PCT_COST)),
               "fixed_sol": K.FIXED_SOL, "rent_sol": K.RENT_SOL, "rent_unrec": K.RENT_UNRECOVERED, "impact_k": K.IMPACT_K, "fill_cap": K.FILL_CAP,
               "off": off, "group": [s["g"] for s in specs], "one_shot": [int(bool(s["one_shot"])) for s in specs], "tok": b64(np.concatenate(tok), np.uint8), "ti": b64(np.concatenate(ti), np.uint16),
               "to": b64(np.concatenate(to), np.uint16), "gross": b64(np.concatenate(gr), np.float32), "pin": b64(pins, np.float32), "eo": b64(np.concatenate(eo), np.uint8),
               "ve": b64(np.concatenate(ve), np.float32), "lcq": b64(q.T, np.int16), "lcbase": [round(float(x), 6) for x in base]}
    json.dump(payload, open(f"{R}/window_payload.json", "w"))
    print("done", f"{time.time()-t0:.0f}s", "groups", len(groups), "candidates", off[-1])

if __name__ == "__main__":
    main()
