"""Multi-fold walk-forward robustness: split the window into K consecutive folds, restart the strategy with the
same starting balance in each fold, and count how many folds are profitable. A strategy is 'robust' only if most
active folds make money, the median fold is positive, and both the in-sample and out-of-sample halves are positive."""
from __future__ import annotations
import numpy as np
from . import engine as E, costs as K

def folds(trades, D, balance, frac, maxpos, k=26, fill_cap=None):
    tok, ti, to, pin, pout, _ = trades
    ve = D["V"][np.maximum(ti - 1, 0), tok]
    T = D["T"]; edges = np.linspace(0, T, k + 1).astype(int)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = E.simulate(tok, ti, to, pin, pout, D["venue"], ve, D["solpx"], int(a), int(b), balance, frac, maxpos, K.PCT_COST,
                       K.FIXED_SOL, K.RENT_SOL, K.RENT_UNRECOVERED, K.IMPACT_K, K.FILL_CAP if fill_cap is None else fill_cap, np.zeros(0))
        out.append((float(m[1]), int(m[4])))
    rets = np.array([r for r, n in out]); n = np.array([n for r, n in out])
    act = n > 0
    return {"folds": k, "active": int(act.sum()), "profitable": int((rets[act] > 0).sum()),
            "share_profitable": float((rets[act] > 0).mean()) if act.any() else 0.0,
            "median_fold_ret": float(np.median(rets[act])) if act.any() else 0.0,
            "worst_fold_ret": float(rets[act].min()) if act.any() else 0.0, "fold_rets": [round(float(x), 4) for x in rets]}

def is_robust(wf, is_ret, oos_ret, min_active=8):
    return bool(wf["active"] >= min_active and wf["share_profitable"] >= 0.6 and wf["median_fold_ret"] > 0 and is_ret > 0 and oos_ret > 0)
