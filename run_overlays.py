"""Stage 2: apply modular agent/check overlays to the best in-sample base strategies of EVERY family.
Overlays: ~30 check masks (SCOUT/FILTER/ANALYST/REGIME questions) x RISK on/off x EXECUTOR on/off."""
from __future__ import annotations
import json, sys, time
import numpy as np
from solstrats import engine as E, costs as K, grid as G, overlay as OV
from run_backtest import prep

# agent-level combos, every single check, and nested "first k checks" sets (0..8 checks)
AGENT_MASKS = {"none": 0, "SCOUT": 1, "FILTER": 30, "ANALYST": 96, "REGIME": 128, "FILTER+ANALYST": 126, "SCOUT+FILTER": 31,
               "SCOUT+FILTER+ANALYST": 127, "ALL CHECKS": 255}
MASKS = sorted(set(list(AGENT_MASKS.values()) + [1 << b for b in range(8)] + [(1 << k) - 1 for k in range(9)] +
                   [30 | 128, 96 | 128, 2 | 4, 16 | 64, 8 | 32]))

def main(res_dir="results", top_per_family=10):
    t0 = time.time()
    meta = json.load(open(f"{res_dir}/sweep_meta.json"))
    sw = np.load(f"{res_dir}/sweep.npz")
    IS = sw["is_"]; combo = sw["cfg_combo"]; size = sw["cfg_size"]; bal = sw["cfg_bal"]
    D = prep(); chk = OV.check_arrays(D)
    combos = meta["combos"]
    sel = (bal == 100.0) & (IS[:, 4] >= 10)
    picks = []
    for fam in E.FAMILIES:
        idx = np.where(sel & np.array([combos[c]["family"] == fam for c in combo]))[0] if sel.any() else []
        if len(idx) == 0: continue
        order = idx[np.argsort(-IS[idx, 3])]
        seen = set()
        for i in order:
            c = int(combo[i])
            if c in seen: continue
            seen.add(c); picks.append((c, int(size[i])))
            if len(seen) >= top_per_family: break
    print("base picks", len(picks), flush=True)
    rows = []; parts = []
    cache = {}
    for c, s in picks:
        cb = combos[c]; fam = cb["family"]
        key = (fam, tuple(cb["p"]))
        if key not in cache:
            hot = D["hot"].get(int(cb["p"][0]), D["hot"][24]) if fam == "narrative" else D["hot"][24]
            cache[key] = E.signals(E.F[fam], np.asarray(cb["p"], float), D["O"], D["H"], D["L"], D["C"], D["V"], D["first"], D["grp"],
                                   D["venue"], D["supply"], D["mint_auth"], D["freeze"], D["top10"], hot, D["theme"], 0, D["T"] - 1)
        base = cache[key]
        for mask in MASKS:
            sig = OV.apply_checks(base, chk, mask)
            for risk in (0, 1):
                e = OV.risk_exit(tuple(cb["e"])) if risk else tuple(cb["e"])
                tr = E.gen_trades(sig, np.asarray(e, float), D["O"], D["H"], D["L"], D["C"], D["last"], D["supply"], cb["one_shot"])
                for ex in (0, 1):
                    rows.append({"base": c, "family": fam, "size": s, "mask": mask, "risk": risk, "executor": ex})
                    parts.append(tr)
    print("overlay combos", len(rows), round(time.time() - t0), "s", flush=True)
    lens = np.array([len(t[0]) for t in parts]); offs = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    tok, ti, to, pin, pout, _ = [np.concatenate([t[k] for t in parts]) for k in range(6)]
    tok = tok.astype(np.int64); ti = ti.astype(np.int64); to = to.astype(np.int64)
    ve = D["V"][np.maximum(ti - 1, 0), tok]
    nb = len(K.BALANCES); n = len(rows)
    cfg_combo = np.repeat(np.arange(n), nb).astype(np.int64)
    cfg_bal = np.tile(K.BALANCES, n)
    fr = np.array([0.08 if r["executor"] else G.SIZING[r["size"]][0] for r in rows])
    mp = np.array([12 if r["executor"] else G.SIZING[r["size"]][1] for r in rows], dtype=np.int64)
    cfg_frac = np.repeat(fr, nb); cfg_mp = np.repeat(mp, nb)
    out = {}
    for name, lo, hi in (("is", 0, D["split"]), ("oos", D["split"], D["T"])):
        r0 = E.simulate_many(tok, ti, to, pin, pout, D["venue"], ve, D["solpx"], offs, cfg_combo, cfg_bal, cfg_frac, cfg_mp, lo, hi,
                             K.PCT_COST, K.FIXED_SOL, K.RENT_SOL, K.RENT_UNRECOVERED, K.IMPACT_K, K.FILL_CAP)
        # EXECUTOR's slippage check: re-run executor rows with a tighter fill cap (1% of bar volume)
        exm = np.repeat(np.array([r["executor"] for r in rows]) == 1, nb)
        r1 = E.simulate_many(tok, ti, to, pin, pout, D["venue"], ve, D["solpx"], offs, cfg_combo[exm], cfg_bal[exm], cfg_frac[exm],
                             cfg_mp[exm], lo, hi, K.PCT_COST, K.FIXED_SOL, K.RENT_SOL, K.RENT_UNRECOVERED, K.IMPACT_K, 0.01)
        r0[exm] = r1
        out[name] = r0
    np.savez_compressed(f"{res_dir}/overlay.npz", is_=out["is"], oos=out["oos"], cfg_combo=cfg_combo, cfg_bal=cfg_bal)
    json.dump({"rows": rows, "n_configs": int(cfg_combo.size)}, open(f"{res_dir}/overlay_meta.json", "w"))
    print("DONE overlay configs", cfg_combo.size, round(time.time() - t0), "s")

if __name__ == "__main__":
    main(*(sys.argv[1:2]))
