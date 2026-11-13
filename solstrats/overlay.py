"""Modular agent / self-check overlays applicable to EVERY strategy family.

Agents: SCOUT (new-pair check), FILTER (mint, freeze, top-10, volume checks), ANALYST (momentum, score gate),
EXECUTOR (8% size cap + tighter slippage/fill check), RISK (hard -6% stop, trail after +12%), LOGGER (records Q&A;
no P&L effect). A check mask selects which of the 7 self-check questions an entry must pass."""
from __future__ import annotations
import numpy as np
from numba import njit, prange
from .engine import desk_score, NONE

CHECKS = ["scout_new_pair", "filter_mint_revoked", "filter_freeze_revoked", "filter_top10_ok", "filter_volume_ok",
          "analyst_momentum_up", "analyst_score_ok", "regime_sol_uptrend"]
AGENT_CHECKS = {"SCOUT": [0], "FILTER": [1, 2, 3, 4], "ANALYST": [5, 6], "REGIME": [7]}
DEFAULTS = {"new_pair_h": 720, "top10_max": 0.2, "vol_min": 5e4, "score_min": 0.6}

@njit(cache=True, parallel=True)
def _score_grid(C, V, first, top10):
    T, N = C.shape
    out = np.zeros((T, N))
    for j in prange(N):
        f0 = first[j]
        if f0 < 0: continue
        for t in range(max(f0 + 7, 7), T):
            if C[t, j] > 0 and C[t - 6, j] > 0:
                out[t, j] = desk_score(C, V, t, j, f0, top10[j])
    return out

def check_arrays(D, new_pair_h=720, top10_max=0.2, vol_min=5e4, score_min=0.6):
    C, V = D["C"], D["V"]; T, N = C.shape
    age = np.arange(T)[:, None] - D["first"][None, :]
    chk = np.zeros((8, T, N), dtype=bool)
    chk[0] = (D["grp"][None, :] == 1) & (age >= 0) & (age <= new_pair_h)
    chk[1] = np.broadcast_to(D["mint_auth"][None, :] == 0, (T, N))
    chk[2] = np.broadcast_to(D["freeze"][None, :] == 0, (T, N))
    t10 = np.nan_to_num(D["top10"], nan=1.0)
    chk[3] = np.broadcast_to(t10[None, :] <= top10_max, (T, N))
    chk[4] = V >= vol_min
    mom = np.zeros((T, N), bool)
    with np.errstate(invalid="ignore"):
        mom[6:] = C[6:] > C[:-6]
    chk[5] = mom
    if "score" not in D:
        D["score"] = _score_grid(C, V, D["first"], D["top10"])
    chk[6] = D["score"] >= score_min
    sol = D["solpx"]; ma = np.full(T, np.inf)
    cs = np.cumsum(sol)
    ma[168:] = (cs[168:] - cs[:-168]) / 168.0
    chk[7] = np.broadcast_to((sol > ma)[:, None], (T, N))   # regime: SOL above its 7-day average
    return chk

def apply_checks(sig, chk, mask):
    if mask == 0:
        return sig
    s = sig.copy()
    for b in range(8):
        if mask & (1 << b):
            s &= chk[b]
    return s

def agents_to_mask(agents):
    m = 0
    for a in agents:
        for b in AGENT_CHECKS.get(a, []):
            m |= 1 << b
    return m

def risk_exit(e):
    tp, sl, act, w, tm, mct, cr = e
    sl2 = min(sl, 0.06)
    act2 = min(act, 0.12)
    w2 = 0.06 if act >= NONE / 2 else min(w, 0.06)
    return (tp, sl2, act2, w2, tm, mct, cr)

def mask_text(mask):
    return ", ".join(CHECKS[b] for b in range(8) if mask & (1 << b)) or "no checks"
