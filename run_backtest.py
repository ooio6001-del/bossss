"""Brute-force sweep: ~4.9M (entry x exit x sizing x balance) combos, each simulated in-sample and out-of-sample."""
from __future__ import annotations
import json, os, sys, time
import numpy as np
from solstrats import data as Dm, engine as E, costs as K, grid as G

def prep(data_dir="data"):
    D = Dm.load(data_dir)
    C = D["C"]; T, N = C.shape
    valid = ~np.isnan(C)
    first = np.where(valid.any(0), valid.argmax(0), -1).astype(np.int64)
    last = np.where(valid.any(0), T - 1 - valid[::-1].argmax(0), -1).astype(np.int64)
    solpx = C[:, D["sol_idx"]].copy() if D["sol_idx"] >= 0 else np.full(T, 80.0)
    m = np.isnan(solpx)
    if m.any():
        solpx[m] = np.nanmean(solpx)
    ntheme = len(Dm.THEME_NAMES)
    hot = {}
    for L in (24, 72, 168):
        r = np.full((T, N), np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            r[L:] = C[L:] / C[:-L] - 1
        h = np.full(T, -1, np.int64)
        for t in range(L, T):
            best, bi = -1e9, -1
            for th in range(ntheme - 1):  # "other" is not a theme
                sel = (D["theme"] == th) & ~np.isnan(r[t]) & (D["grp"] != 2)
                if sel.sum() >= 2:
                    v = np.nanmean(r[t, sel])
                    if v > best: best, bi = v, th
            h[t] = bi
        hot[L] = h
    D.update(first=first, last=last, solpx=solpx, hot=hot, T=T, N=N, split=int(T * 0.6))
    return D

def run_combo_trades(D, fam, p, exits, one_shot):
    fi = E.F[fam]
    hot = D["hot"].get(int(p[0]), D["hot"][24]) if fam == "narrative" else D["hot"][24]
    sig = E.signals(fi, np.asarray(p, float), D["O"], D["H"], D["L"], D["C"], D["V"], D["first"], D["grp"], D["venue"],
                    D["supply"], D["mint_auth"], D["freeze"], D["top10"], hot, D["theme"], 0, D["T"] - 1)
    out = []
    for e in exits:
        out.append(E.gen_trades(sig, np.asarray(e, float), D["O"], D["H"], D["L"], D["C"], D["last"], D["supply"], one_shot))
    return sig, out

def main(out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    D = prep()
    print("data", D["T"], "hours x", D["N"], "tokens; split at bar", D["split"], flush=True)
    combos = []; parts = []
    for fam, p, es, one in G.entries():
        exits = G.EXIT_SETS[es]
        _, trs = run_combo_trades(D, fam, p, exits, one)
        for e, tr in zip(exits, trs):
            combos.append({"family": fam, "p": list(p), "e": list(e), "one_shot": one})
            parts.append(tr)
    print("combos", len(combos), "trade gen", round(time.time() - t0), "s", flush=True)
    lens = np.array([len(t[0]) for t in parts]); offs = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    tok, ti, to, pin, pout, rs = [np.concatenate([t[k] for t in parts]) for k in range(6)]
    tok = tok.astype(np.int64); ti = ti.astype(np.int64); to = to.astype(np.int64)
    vol_entry = D["V"][np.maximum(ti - 1, 0), tok]   # USD volume of the signal bar (known at entry)
    np.savez_compressed(f"{out_dir}/trades.npz", tok=tok, ti=ti, to=to, pin=pin, pout=pout, rs=rs, offs=offs)
    nb = len(K.BALANCES); ns = len(G.SIZING); nc = len(combos)
    cfg_combo = np.repeat(np.arange(nc), ns * nb).astype(np.int64)
    cfg_size = np.tile(np.repeat(np.arange(ns), nb), nc)
    cfg_bal = np.tile(K.BALANCES, nc * ns)
    cfg_frac = np.array([G.SIZING[s][0] for s in range(ns)])[cfg_size]
    cfg_maxpos = np.array([G.SIZING[s][1] for s in range(ns)], dtype=np.int64)[cfg_size]
    print("configs", cfg_combo.size, flush=True)
    args = (K.PCT_COST, K.FIXED_SOL, K.RENT_SOL, K.RENT_UNRECOVERED, K.IMPACT_K, K.FILL_CAP)
    res = {}
    for name, lo, hi in (("is", 0, D["split"]), ("oos", D["split"], D["T"]), ("full", 0, D["T"])):
        t1 = time.time()
        res[name] = E.simulate_many(tok, ti, to, pin, pout, D["venue"], vol_entry, D["solpx"], offs, cfg_combo, cfg_bal,
                                    cfg_frac, cfg_maxpos, lo, hi, *args)
        print(name, "sim", round(time.time() - t1), "s", flush=True)
    np.savez_compressed(f"{out_dir}/sweep.npz", is_=res["is"], oos=res["oos"], full=res["full"], cfg_combo=cfg_combo,
                        cfg_size=cfg_size, cfg_bal=cfg_bal)
    meta = {"combos": combos, "sizing": G.SIZING, "balances": K.BALANCES.tolist(), "n_configs": int(cfg_combo.size),
            "T": D["T"], "N": D["N"], "split": D["split"], "t0": float(D["times"][0]), "t_end": float(D["times"][-1]),
            "t_split": float(D["times"][D["split"]]), "metrics": E.M_NAMES, "space_size": G.space_size(),
            "n_trades_total": int(lens.sum()), "elapsed_s": round(time.time() - t0)}
    json.dump(meta, open(f"{out_dir}/sweep_meta.json", "w"))
    print("DONE", meta["n_configs"], "configs in", meta["elapsed_s"], "s", flush=True)

if __name__ == "__main__":
    main(*(sys.argv[1:2]))
