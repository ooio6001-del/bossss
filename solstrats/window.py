"""Date-range (window) recompute: positions start flat at the window start, same cost model as engine.simulate,
open positions at the window end are marked to the last close of the window (unrealized P&L, exit costs included).

Exact path (Python / API): signals before the window are cleared, trades are regenerated (so a pre-window position
can never block an in-window entry), then simulate_window runs on [t_lo, t_hi).
"""
from __future__ import annotations
import math, time
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from numba import njit
from . import costs as K, engine as E

CHI = ZoneInfo("America/Chicago")
PRESETS_H = {"1h": 1, "4h": 4, "12h": 12, "24h": 24, "48h": 48, "7d": 168, "30d": 720, "90d": 2160}
W_NAMES = ["final", "ret", "maxdd", "ntrades", "wins", "realized", "n_open", "unrealized", "fees", "skipped", "winrate", "profit"]

@njit(cache=True)
def simulate_window(tok, ti, to, pin, pout, isopen, venue_of_tok, vol_entry, solpx, t_lo, t_hi, balance, frac, maxpos,
                    pct_cost, fixed_sol, rent_sol, rent_unrec, impact_k, fill_cap):
    """Port of engine.simulate with open-at-end trades (isopen=1) booked as unrealized instead of closed trades."""
    out = np.zeros(12)
    n = tok.size
    order = np.argsort(ti * 1.0 + tok * 1e-7)
    cash = balance
    op_exit = np.full(maxpos, -1, np.int64); op_val = np.zeros(maxpos); op_cost = np.zeros(maxpos)
    op_lock = np.zeros(maxpos); op_open = np.zeros(maxpos, np.int64)
    nopen = 0; peak = balance; maxdd = 0.0
    ntr = 0; wins = 0; realized = 0.0; nunr = 0; unreal = 0.0; skipped = 0; fees = 0.0
    for ii in range(n + 1):
        if ii < n:
            q = order[ii]; t_now = ti[q]
            if t_now < t_lo or t_now >= t_hi:
                continue
        else:
            t_now = 1 << 40
        while True:
            mi = -1; me = 1 << 41
            for s in range(maxpos):
                if op_exit[s] >= 0 and op_exit[s] < t_now and op_exit[s] < me:
                    me = op_exit[s]; mi = s
            if mi < 0: break
            cash += op_val[mi] + op_lock[mi] * (1 - rent_unrec)
            pnl = op_val[mi] - op_cost[mi] - op_lock[mi] * rent_unrec
            if op_open[mi] == 1:
                nunr += 1; unreal += pnl
            else:
                ntr += 1; realized += pnl
                if pnl > 0: wins += 1
            op_exit[mi] = -1; nopen -= 1
            eq = cash
            for s in range(maxpos):
                if op_exit[s] >= 0: eq += op_cost[s] + op_lock[s]
            if eq > peak: peak = eq
            dd = 1 - eq / peak if peak > 0 else 0.0
            if dd > maxdd: maxdd = dd
        if ii >= n: break
        if nopen >= maxpos:
            skipped += 1; continue
        eq = cash
        for s in range(maxpos):
            if op_exit[s] >= 0: eq += op_cost[s] + op_lock[s]
        fixed_mode = frac < 0
        size = -frac if fixed_mode else frac * eq
        sp = solpx[t_now]; fix_usd = fixed_sol * sp; lock_usd = rent_sol * sp
        if size + lock_usd > cash:
            if fixed_mode:
                skipped += 1; continue
            size = cash - lock_usd
        if size <= (0.0 if fixed_mode else fix_usd * 2):
            skipped += 1; continue
        ve = vol_entry[q]
        if size > fill_cap * ve:
            skipped += 1; continue
        pc = pct_cost[venue_of_tok[tok[q]]] + impact_k * size / (ve + 1.0)
        gross = pout[q] / pin[q]
        val = (size - fix_usd) * (1 - pc) * gross * (1 - pc) - fix_usd
        fees += size * pc * (1 + gross) + 2 * fix_usd + rent_unrec * lock_usd
        cash -= size + lock_usd
        for s in range(maxpos):
            if op_exit[s] < 0:
                op_exit[s] = to[q]; op_val[s] = val; op_cost[s] = size; op_lock[s] = lock_usd; op_open[s] = isopen[q]
                break
        nopen += 1
    out[0] = cash; out[1] = cash / balance - 1.0; out[2] = maxdd; out[3] = ntr; out[4] = wins; out[5] = realized
    out[6] = nunr; out[7] = unreal; out[8] = fees; out[9] = skipped; out[10] = wins / ntr if ntr > 0 else 0.0
    out[11] = cash - balance
    return out

def end_open(D, tr):
    """True where the engine closed a trade only because the data ended while the token was still trading."""
    tok, to = np.asarray(tr[0]), np.asarray(tr[2])
    rs = np.asarray(tr[5]) if len(tr) > 5 else np.full(tok.size, -1)
    return (rs == E.R_END) & (to == D["last"][tok]) & (D["last"][tok] >= int(D["T"]) - 1)

def clip_to_window(D, tr, t_lo, t_hi):
    """Keep trades entering in [t_lo,t_hi); trades still open at t_hi-1 are marked at that bar's close."""
    tok, ti, to, pin, pout = [np.asarray(x) for x in tr[:5]]
    eo = end_open(D, tr)
    m = (ti >= t_lo) & (ti < t_hi)
    tok, ti, to, pin, pout, eo = tok[m], ti[m], to[m].copy(), pin[m], pout[m].copy(), eo[m]
    # open at window end: exits after the window, or the engine's forced end-of-data exit on a still-trading token
    isopen = ((to >= t_hi) | (eo & (to == t_hi - 1))).astype(np.int64)
    if isopen.any():
        k = np.nonzero(isopen)[0]
        mark = D["C"][t_hi - 1, tok[k]]
        mark = np.where(np.isfinite(mark) & (mark > 0), mark, pin[k])
        pout[k] = mark; to[k] = t_hi - 1
    return tok, ti, to, pin, pout, isopen

def run(D, tr, t_lo, t_hi, bal, frac, maxpos, fill_cap=K.FILL_CAP):
    tok, ti, to, pin, pout, isopen = clip_to_window(D, tr, t_lo, t_hi)
    ve = D["V"][np.maximum(ti - 1, 0), tok] if tok.size else np.zeros(0)
    return simulate_window(tok.astype(np.int64), ti.astype(np.int64), to.astype(np.int64), pin.astype(float), pout.astype(float),
                           isopen, D["venue"], ve.astype(float), D["solpx"], int(t_lo), int(t_hi), float(bal), float(frac),
                           int(maxpos), K.PCT_COST, K.FIXED_SOL, K.RENT_SOL, K.RENT_UNRECOVERED, K.IMPACT_K, fill_cap)

def pack(m, bal):
    d = {n: float(m[i]) for i, n in enumerate(W_NAMES)}
    for k in ("ntrades", "wins", "n_open", "skipped"): d[k] = int(d[k])
    d["start"] = float(bal); d["equity_end"] = d["final"]
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}

def resolve(D, preset=None, start=None, end=None):
    """Window bounds as bar indices [t_lo, t_hi). start/end: unix seconds or ISO strings (hour precision, end inclusive).
    YTD and 12m are capped to the available hourly span; the returned dict says so."""
    times = D["times"]; T = int(D["T"])
    def to_ts(x):
        if x is None: return None
        if isinstance(x, (int, float)) or str(x).lstrip("-").isdigit(): return float(x)
        dt = datetime.fromisoformat(str(x))
        if dt.tzinfo is None: dt = dt.replace(tzinfo=CHI)
        return dt.timestamp()
    t_hi = T; capped = False; label = preset or "custom"
    if preset in PRESETS_H:
        n = PRESETS_H[preset]; t_lo = max(0, T - n); capped = T < n
    elif preset in ("ytd", "YTD", "12m"):
        end_dt = datetime.fromtimestamp(float(times[-1]), CHI)
        if preset == "12m":
            want = float(times[-1]) - 365 * 86400 + 3600
        else:
            want = datetime(end_dt.year, 1, 1, tzinfo=CHI).timestamp()
        t_lo = int(np.searchsorted(times, want)); capped = want < times[0]
        label = preset.upper() if preset != "12m" else "12m"
    else:
        s, e = to_ts(start), to_ts(end)
        t_lo = 0 if s is None else int(np.searchsorted(times, s))
        t_hi = T if e is None else int(np.searchsorted(times, e, side="right"))
        capped = (s is not None and s < times[0]) or (e is not None and e > times[-1])
    t_lo = max(0, min(t_lo, T - 1)); t_hi = max(t_lo + 1, min(t_hi, T))
    f = lambda t: datetime.fromtimestamp(float(t), CHI).strftime("%Y-%m-%d %H:%M %Z")
    return {"t_lo": t_lo, "t_hi": t_hi, "hours": t_hi - t_lo, "label": label, "capped_to_data": bool(capped),
            "start": f(times[t_lo]), "end": f(times[t_hi - 1]), "data_start": f(times[0]), "data_end": f(times[-1])}

def replay(cand, t_lo, one_shot):
    """Rebuild gen_trades(signal masked before the window) from gen_trades_all candidates: per token, the first candidate
    filling at or after t_lo, then the next filling at or after previous exit + 2 (the engine resumes scanning signals
    at exit+1, which fill one bar later). Candidates must be sorted by (token, fill bar)."""
    tok, ti, to = cand[0], cand[1], cand[2]
    keep = np.zeros(tok.size, np.bool_)
    _replay(tok, ti, to, int(t_lo), bool(one_shot), keep)
    return tuple(np.asarray(x)[keep] for x in cand)

@njit(cache=True)
def _replay(tok, ti, to, t_lo, one_shot, keep):
    cur = -1; nxt = 0; done = False
    for q in range(tok.size):
        if tok[q] != cur:
            cur = tok[q]; nxt = t_lo; done = False
        if done or ti[q] < nxt:
            continue
        keep[q] = True; nxt = to[q] + 2
        if one_shot:
            done = True
