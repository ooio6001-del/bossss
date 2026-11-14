#!/usr/bin/env python3
"""Batch 8 hunter: 10,000 NEW variants, mostly fresh families, frozen holdout.

Shifts away from the mined Double-N neighborhood. All 80 entries are new
families (TRIX, Coppock, KST, Awesome Oscillator, Ichimoku cloud, Stochastic
RSI, Vortex, TSI). Ranking uses the same frozen holdout (bars after 3961).
If a newer complete hourly bar can be fetched, it is evaluated separately
as a truly unseen holdout without modifying the main data.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from numba import njit, prange

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from solstrats import costs as K
from solstrats import engine as E
from solstrats.describe import exit_text, sizing_text

import hunt as H
import hunt_batch3 as H3
import hunt_batch5 as H5
import hunt_batch6 as H6
import hunt_batch7 as H7
import wf_lib as WF

DATA_DIR = H.DATA_DIR
BATCH_SIZE = 10_000
NONE = H.NONE
CHI = ZoneInfo("America/Chicago")
FROZEN_HOLDOUT_LO = 3961
FROZEN_T = 4297  # last bar used by batches 1–7

EXITS = [
    (0.22, 0.17, NONE, 0.0, 42, NONE, 1.0),
    (0.65, 0.23, 0.32, 0.15, 156, NONE, 1.0),
    (1.25, 0.29, NONE, 0.0, 276, NONE, 1.0),
    (NONE, 0.27, 0.48, 0.19, 396, NONE, 1.0),
    (1.80, 0.20, 0.44, 0.16, 228, NONE, 1.0),
]
SIZING = [(0.055, 18), (0.115, 8), (0.165, 6), (0.27, 3), (0.36, 2)]
BALANCES = np.array([48.0, 72.0, 125.0, 215.0, 340.0], dtype=np.float64)


def entry_text(fam, p):
    if fam == "trix_cross":
        n, sig, mode = int(p[0]), int(p[1]), int(p[2])
        if mode == 0:
            return f"Buy when TRIX({n}) crosses up through zero"
        if mode == 1:
            return f"Buy when TRIX({n}) crosses up through its {sig}-bar signal"
        return f"Buy when TRIX({n}) crosses up through its {sig}-bar signal while still below zero"
    if fam == "coppock_turn":
        a, b, w, mode = int(p[0]), int(p[1]), int(p[2]), int(p[3])
        if mode == 0:
            return f"Buy when the Coppock curve (ROC {a}+{b}, WMA {w}) crosses up through zero"
        return f"Buy when the Coppock curve (ROC {a}+{b}, WMA {w}) turns up while below zero"
    if fam == "kst_cross":
        return f"Buy when KST({int(p[0])}-scale) crosses up through its {int(p[1])}-bar signal"
    if fam == "awesome_osc":
        f, s, mode = int(p[0]), int(p[1]), int(p[2])
        if mode == 0:
            return f"Buy when the Awesome Oscillator ({f}/{s}) crosses up through zero"
        return f"Buy an Awesome Oscillator ({f}/{s}) saucer while the histogram is still below zero"
    if fam == "ichimoku_cloud":
        ten, kij, sb, mode = int(p[0]), int(p[1]), int(p[2]), int(p[3])
        cloud = f"Ichimoku cloud ({ten}/{kij}/{sb})"
        if mode == 0:
            return f"Buy when close crosses up through the {cloud}"
        if mode == 1:
            return f"Buy when Tenkan crosses up through Kijun while close is above the {cloud}"
        return f"Buy when close is above the {cloud} and Tenkan is above Kijun, first bar of that state"
    if fam == "stoch_rsi":
        return (
            f"Buy when Stochastic RSI({int(p[0])},{int(p[1])}) crosses up from below {p[3]:g} "
            f"(D smoothed {int(p[2])})"
        )
    if fam == "vortex_cross":
        return f"Buy when Vortex VI+ crosses up through VI− (period {int(p[0])})"
    if fam == "tsi_cross":
        r, s, sig, mode = int(p[0]), int(p[1]), int(p[2]), int(p[3])
        if mode == 0:
            return f"Buy when TSI({r},{s}) crosses up through its {sig}-bar signal"
        return f"Buy when TSI({r},{s}) crosses up through its {sig}-bar signal while still below zero"
    return f"{fam} {p}"


def entries():
    Elist = []
    for p in (
        (12, 9, 0), (12, 9, 1), (12, 9, 2), (15, 9, 1), (9, 5, 0),
        (9, 5, 1), (18, 9, 1), (21, 7, 0), (15, 5, 2), (8, 5, 1),
    ):
        Elist.append(("trix_cross", p))
    for p in (
        (11, 14, 10, 0), (12, 24, 10, 0), (24, 48, 10, 0), (11, 14, 10, 1),
        (24, 48, 14, 0), (16, 32, 10, 0), (8, 16, 8, 0), (24, 48, 10, 1),
        (20, 40, 12, 0), (14, 28, 10, 0),
    ):
        Elist.append(("coppock_turn", p))
    for p in (
        (1, 9), (1, 6), (2, 9), (2, 12), (3, 9),
        (1, 15), (2, 6), (3, 12), (1, 7), (2, 15),
    ):
        Elist.append(("kst_cross", p))
    for p in (
        (5, 34, 0), (5, 34, 1), (7, 28, 0), (7, 28, 1), (5, 21, 0),
        (5, 21, 1), (8, 34, 0), (6, 40, 1), (5, 48, 0), (9, 34, 1),
    ):
        Elist.append(("awesome_osc", p))
    for p in (
        (9, 26, 52, 0), (9, 26, 52, 1), (7, 22, 44, 0), (7, 22, 44, 1),
        (12, 30, 60, 0), (10, 30, 60, 1), (9, 26, 52, 2), (8, 22, 44, 0),
        (9, 24, 48, 0), (11, 32, 64, 1),
    ):
        Elist.append(("ichimoku_cloud", p))
    for p in (
        (14, 14, 3, 0.20), (14, 14, 3, 0.15), (10, 10, 3, 0.20), (21, 14, 3, 0.20),
        (14, 10, 3, 0.20), (7, 7, 3, 0.20), (14, 14, 5, 0.20), (16, 16, 3, 0.10),
        (21, 21, 3, 0.20), (10, 14, 3, 0.25),
    ):
        Elist.append(("stoch_rsi", p))
    for p in ((14,), (21,), (10,), (28,), (7,), (16,), (24,), (32,), (18,), (12,)):
        Elist.append(("vortex_cross", p))
    for p in (
        (25, 13, 7, 0), (25, 13, 7, 1), (13, 7, 5, 0), (13, 7, 5, 1),
        (20, 10, 5, 0), (30, 15, 7, 1), (8, 5, 3, 0), (25, 13, 9, 0),
        (15, 8, 5, 1), (40, 20, 7, 0),
    ):
        Elist.append(("tsi_cross", p))
    if len(Elist) != 80:
        raise RuntimeError(f"expected 80 entries, got {len(Elist)}")
    return Elist


def ema_any(X, L):
    """EMA over any finite values, including zero and negatives (unlike H3.ema)."""
    T, N = X.shape
    out = np.full((T, N), np.nan, dtype=np.float64)
    a = 2.0 / (L + 1.0)
    for j in range(N):
        started = False
        prev = 0.0
        for t in range(T):
            v = X[t, j]
            if not (v == v):
                if started:
                    out[t, j] = prev
                continue
            if not started:
                prev = v
                out[t, j] = prev
                started = True
            else:
                prev = a * v + (1.0 - a) * prev
                out[t, j] = prev
    return out


def roc(C, n):
    out = np.full(C.shape, np.nan, dtype=np.float64)
    n = int(n)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[n:] = C[n:] / C[:-n] - 1.0
        out[n:][~(C[:-n] > 0)] = np.nan
    return out


def trix_grid(C, n):
    e1 = H3.ema(C, n)
    e2 = H3.ema(e1, n)
    e3 = H3.ema(e2, n)
    out = np.full(C.shape, np.nan, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[1:] = e3[1:] / e3[:-1] - 1.0
        out[1:][~(e3[:-1] == e3[:-1]) | ~(e3[:-1] != 0)] = np.nan
    return out


def kst_grid(C, scale):
    # Classic ROC 10,15,20,30 and SMA 10,10,10,15, scaled for hourly.
    r1, r2, r3, r4 = 10 * scale, 15 * scale, 20 * scale, 30 * scale
    s1, s2, s3, s4 = 10 * scale, 10 * scale, 10 * scale, 15 * scale
    rc1, rc2, rc3, rc4 = roc(C, r1), roc(C, r2), roc(C, r3), roc(C, r4)
    k = (
        1.0 * H.rolling_mean(rc1, s1)
        + 2.0 * H.rolling_mean(rc2, s2)
        + 3.0 * H.rolling_mean(rc3, s3)
        + 4.0 * H.rolling_mean(rc4, s4)
    )
    return k


def tsi_grid(C, r, s):
    mom = np.zeros_like(C, dtype=np.float64)
    mom[1:] = C[1:] - C[:-1]
    abs_m = np.abs(mom)
    num = ema_any(ema_any(mom, r), s)
    den = ema_any(ema_any(abs_m, r), s)
    out = np.full(C.shape, np.nan, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = 100.0 * num / den
        out[~(den != 0)] = np.nan
    return out


def stoch_rsi_grid(C, rsi_n, stoch_n, d_n):
    rsi = H5.rsi_grid(C, rsi_n)
    lo = H5.rolling_min(rsi, stoch_n)
    hi = H6.rolling_max(rsi, stoch_n)
    k = np.full(C.shape, np.nan, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        k = (rsi - lo) / (hi - lo)
        k[~(hi > lo)] = np.nan
    if d_n > 1:
        k = H.rolling_mean(k, d_n)
    return k


def ichimoku_parts(H_, L_, ten, kij, sb):
    tenkan = 0.5 * (H6.rolling_max(H_, ten) + H5.rolling_min(L_, ten))
    kijun = 0.5 * (H6.rolling_max(H_, kij) + H5.rolling_min(L_, kij))
    raw_a = 0.5 * (tenkan + kijun)
    raw_b = 0.5 * (H6.rolling_max(H_, sb) + H5.rolling_min(L_, sb))
    # Displace forward by kijun bars: cloud at t uses values from t-kijun (no lookahead)
    span_a = np.full(H_.shape, np.nan, dtype=np.float64)
    span_b = np.full(H_.shape, np.nan, dtype=np.float64)
    span_a[kij:] = raw_a[:-kij]
    span_b[kij:] = raw_b[:-kij]
    return tenkan, kijun, span_a, span_b


def vortex_grid(H_, L_, C, n):
    tr = H3.true_range(H_, L_, C)
    vm_p = np.zeros_like(C, dtype=np.float64)
    vm_m = np.zeros_like(C, dtype=np.float64)
    vm_p[1:] = np.abs(H_[1:] - L_[:-1])
    vm_m[1:] = np.abs(L_[1:] - H_[:-1])
    s_tr = H.rolling_mean(tr, n) * n
    s_p = H.rolling_mean(vm_p, n) * n
    s_m = H.rolling_mean(vm_m, n) * n
    with np.errstate(invalid="ignore", divide="ignore"):
        vip = s_p / s_tr
        vim = s_m / s_tr
        vip[~(s_tr > 0)] = np.nan
        vim[~(s_tr > 0)] = np.nan
    return vip, vim


@njit(cache=True, parallel=True)
def sig_cross_level(ind, first, grp, level, t_hi):
    T, N = ind.shape
    out = np.zeros((T, N), dtype=np.bool_)
    for j in prange(N):
        if grp[j] == 2 or first[j] < 0:
            continue
        f0 = int(first[j])
        start = max(f0 + 2, 2)
        for t in range(start, t_hi):
            v, v1 = ind[t, j], ind[t - 1, j]
            if v == v and v1 == v1 and v1 <= level and v > level:
                out[t, j] = True
    return out


@njit(cache=True, parallel=True)
def sig_cross_up_line(fast, slow, first, grp, t_hi, only_below=False):
    T, N = fast.shape
    out = np.zeros((T, N), dtype=np.bool_)
    for j in prange(N):
        if grp[j] == 2 or first[j] < 0:
            continue
        f0 = int(first[j])
        start = max(f0 + 2, 2)
        for t in range(start, t_hi):
            f, f1, s, s1 = fast[t, j], fast[t - 1, j], slow[t, j], slow[t - 1, j]
            if not (f == f and f1 == f1 and s == s and s1 == s1):
                continue
            if f > s and f1 <= s1:
                if only_below and not (f < 0.0):
                    continue
                out[t, j] = True
    return out


@njit(cache=True, parallel=True)
def sig_turn_up_below(ind, first, grp, t_hi):
    T, N = ind.shape
    out = np.zeros((T, N), dtype=np.bool_)
    for j in prange(N):
        if grp[j] == 2 or first[j] < 0:
            continue
        f0 = int(first[j])
        start = max(f0 + 3, 3)
        for t in range(start, t_hi):
            v, v1, v2 = ind[t, j], ind[t - 1, j], ind[t - 2, j]
            if v == v and v1 == v1 and v2 == v2 and v < 0.0 and v > v1 and v1 <= v2:
                out[t, j] = True
    return out


@njit(cache=True, parallel=True)
def sig_ao_saucer(ao, first, grp, t_hi):
    T, N = ao.shape
    out = np.zeros((T, N), dtype=np.bool_)
    for j in prange(N):
        if grp[j] == 2 or first[j] < 0:
            continue
        f0 = int(first[j])
        start = max(f0 + 3, 3)
        for t in range(start, t_hi):
            a, a1, a2 = ao[t, j], ao[t - 1, j], ao[t - 2, j]
            if not (a == a and a1 == a1 and a2 == a2):
                continue
            if a < 0.0 and a1 < 0.0 and a2 < 0.0 and a2 >= a1 and a > a1:
                out[t, j] = True
    return out


@njit(cache=True, parallel=True)
def sig_ichimoku(C, tenkan, kijun, span_a, span_b, first, grp, mode, t_hi):
    T, N = C.shape
    out = np.zeros((T, N), dtype=np.bool_)
    for j in prange(N):
        if grp[j] == 2 or first[j] < 0:
            continue
        f0 = int(first[j])
        start = max(f0 + 3, 3)
        prev = False
        for t in range(start, t_hi):
            c, c1 = C[t, j], C[t - 1, j]
            a, b, a1, b1 = span_a[t, j], span_b[t, j], span_a[t - 1, j], span_b[t - 1, j]
            tn, kn, tn1, kn1 = tenkan[t, j], kijun[t, j], tenkan[t - 1, j], kijun[t - 1, j]
            if not (c == c and c1 == c1 and a == a and b == b and a1 == a1 and b1 == b1):
                prev = False
                continue
            top = a if a > b else b
            top1 = a1 if a1 > b1 else b1
            above = c > top
            if mode == 0:
                if c > top and c1 <= top1:
                    out[t, j] = True
            elif mode == 1:
                if not (tn == tn and kn == kn and tn1 == tn1 and kn1 == kn1):
                    continue
                if above and tn > kn and tn1 <= kn1:
                    out[t, j] = True
            else:
                if not (tn == tn and kn == kn):
                    prev = False
                    continue
                state = above and tn > kn
                if state and not prev:
                    out[t, j] = True
                prev = state
    return out


def make_signal(fam, p, D, cache):
    C, H_, L_ = D["C"], D["H"], D["L"]
    first, grp = D["first"], D["grp"]
    t_hi = D["T"] - 1
    if fam == "trix_cross":
        n, sigl, mode = int(p[0]), int(p[1]), int(p[2])
        tx = cache.setdefault(("trix", n), trix_grid(C, n))
        if mode == 0:
            return sig_cross_level(tx, first, grp, 0.0, t_hi)
        sig = cache.setdefault(("trix_sig", n, sigl), ema_any(tx, sigl))
        return sig_cross_up_line(tx, sig, first, grp, t_hi, only_below=(mode == 2))
    if fam == "coppock_turn":
        a, b, w, mode = int(p[0]), int(p[1]), int(p[2]), int(p[3])
        key = ("coppock", a, b, w)
        if key not in cache:
            cache[key] = H6.wma(roc(C, a) + roc(C, b), w)
        cp = cache[key]
        if mode == 0:
            return sig_cross_level(cp, first, grp, 0.0, t_hi)
        return sig_turn_up_below(cp, first, grp, t_hi)
    if fam == "kst_cross":
        scale, sigl = int(p[0]), int(p[1])
        k = cache.setdefault(("kst", scale), kst_grid(C, scale))
        sig = cache.setdefault(("kst_sig", scale, sigl), H.rolling_mean(k, sigl))
        return sig_cross_up_line(k, sig, first, grp, t_hi, False)
    if fam == "awesome_osc":
        f, s, mode = int(p[0]), int(p[1]), int(p[2])
        key = ("ao", f, s)
        if key not in cache:
            mid = 0.5 * (H_ + L_)
            cache[key] = H.rolling_mean(mid, f) - H.rolling_mean(mid, s)
        ao = cache[key]
        if mode == 0:
            return sig_cross_level(ao, first, grp, 0.0, t_hi)
        return sig_ao_saucer(ao, first, grp, t_hi)
    if fam == "ichimoku_cloud":
        ten, kij, sb, mode = int(p[0]), int(p[1]), int(p[2]), int(p[3])
        key = ("ichi", ten, kij, sb)
        if key not in cache:
            cache[key] = ichimoku_parts(H_, L_, ten, kij, sb)
        tenkan, kijun, spa, spb = cache[key]
        return sig_ichimoku(C, tenkan, kijun, spa, spb, first, grp, mode, t_hi)
    if fam == "stoch_rsi":
        rn, sn, dn, thr = int(p[0]), int(p[1]), int(p[2]), float(p[3])
        k = cache.setdefault(("stochrsi", rn, sn, dn), stoch_rsi_grid(C, rn, sn, dn))
        return sig_cross_level(k, first, grp, thr, t_hi)
    if fam == "vortex_cross":
        n = int(p[0])
        key = ("vortex", n)
        if key not in cache:
            cache[key] = vortex_grid(H_, L_, C, n)
        vip, vim = cache[key]
        return sig_cross_up_line(vip, vim, first, grp, t_hi, False)
    if fam == "tsi_cross":
        r, s, sigl, mode = int(p[0]), int(p[1]), int(p[2]), int(p[3])
        ts = cache.setdefault(("tsi", r, s), tsi_grid(C, r, s))
        sig = cache.setdefault(("tsi_sig", r, s, sigl), ema_any(ts, sigl))
        return sig_cross_up_line(ts, sig, first, grp, t_hi, only_below=(mode == 1))
    raise KeyError(fam)


def row_from(i, batch, fam, p, e, frac, maxpos, bal, oos, is_, full, hold, full_all):
    oos_profit = float(oos[0] - bal)
    is_ret = float(is_[1])
    hold_profit = float(hold[0] - bal)
    return {
        "id": f"H{batch}-{i:05d}",
        "batch": batch,
        "family": fam,
        "entry": entry_text(fam, p),
        "exit": exit_text(e),
        "sizing": sizing_text(frac, maxpos),
        "balance": bal,
        "oos_end": round(float(oos[0]), 4),
        "oos_profit": round(oos_profit, 4),
        "oos_ret": round(float(oos[1]), 4),
        "oos_win": round(float(oos[13]), 4),
        "oos_trades": int(oos[4]),
        "oos_dd": round(float(oos[2]), 4),
        "oos_sharpe": round(float(oos[3]), 4),
        "oos_pf": round(float(oos[12]), 4),
        "oos_avg_win": round(float(oos[7]), 6),
        "oos_avg_loss": round(float(oos[8]), 6),
        "is_end": round(float(is_[0]), 4),
        "is_profit": round(float(is_[0] - bal), 4),
        "is_ret": round(is_ret, 4),
        "is_sharpe": round(float(is_[3]), 4),
        "is_trades": int(is_[4]),
        "is_win": round(float(is_[13]), 4),
        "is_dd": round(float(is_[2]), 4),
        "full_end": round(float(full[0]), 4),
        "full_ret": round(float(full[1]), 4),
        "full_win": round(float(full[13]), 4),
        "full_trades": int(full[4]),
        "full_dd": round(float(full[2]), 4),
        "full_fees": round(float(full[11]), 4),
        "both_windows": H.both_windows_flag(oos_profit, is_ret),
        "params": json.dumps(list(map(float, p))),
        "exit_params": json.dumps(list(map(float, e))),
        "holdout_end": round(float(hold[0]), 4),
        "holdout_profit": round(hold_profit, 4),
        "holdout_ret": round(float(hold[1]), 4),
        "holdout_trades": int(hold[4]),
        "holdout_win": round(float(hold[13]), 4),
        "holdout_dd": round(float(hold[2]), 4),
        "holdout_sharpe": round(float(hold[3]), 4),
        "holdout_pf": round(float(hold[12]), 4),
        "full_all_ret": round(float(full_all[1]), 4),
        "full_all_trades": int(full_all[4]),
    }


def smoke_signals(D, cache):
    problems, counts = [], {}
    for fam, p in entries():
        sig = make_signal(fam, p, D, cache)
        if sig.shape != (D["T"], D["N"]):
            problems.append(f"{fam} shape {sig.shape}")
            continue
        n = int(sig.sum())
        counts[fam] = counts.get(fam, 0) + n
        if n == 0:
            problems.append(f"zero signals: {entry_text(fam, p)}")
    return problems, counts


def probe_new_hourly_bars():
    """One GeckoTerminal request. Does not write into the main data dir."""
    from datetime import datetime, timezone
    from solstrats.http import Client

    result = {
        "probed": True,
        "new_complete_bars": 0,
        "newest_candle": None,
        "note": "",
    }
    try:
        # Use a pool known to be in the aligned dataset (from batch 6 last bar).
        # SOL-ish or any file ending at the grid last ts.
        import json as _json
        from pathlib import Path as _P

        pool = None
        last_grid = None
        for fn in (Path(__file__).resolve().parent.parent / "run" / "data" / "ohlcv").glob("*.json"):
            d = _json.loads(fn.read_text())
            if d.get("ohlcv") and d["ohlcv"][-1][0] >= 1794603600:
                pool = d["meta"]["pool"]
                last_grid = d["ohlcv"][-1][0]
                break
        if not pool:
            result["note"] = "could not find a current-universe pool to probe"
            return result
        c = Client(cache_dir=str(ROOT / "cache" / "gt_probe"), base_delay=2.0)
        url = f"https://api.geckoterminal.com/api/v2/networks/solana/pools/{pool}/ohlcv/hour"
        r = c.get("geckoterminal", url, params={"aggregate": 1, "limit": 5, "currency": "usd", "token": "base"})
        rows = r["data"]["attributes"]["ohlcv_list"]
        newest = max(row[0] for row in rows) if rows else None
        now = time.time()
        if newest is not None:
            result["newest_candle"] = datetime.fromtimestamp(float(newest), tz=timezone.utc).astimezone(CHI).strftime(
                "%Y-%m-%d %I:%M %p %Z"
            )
            # A complete new hour must have opened after the frozen last bar AND be fully closed.
            if newest > last_grid and now >= newest + 3600:
                result["new_complete_bars"] = 1
                result["note"] = "a newer complete hourly bar is available"
            elif newest > last_grid:
                result["note"] = (
                    f"API shows a bar opening at {result['newest_candle']}, but it is not complete yet"
                )
            else:
                result["note"] = (
                    f"newest GeckoTerminal candle is still {result['newest_candle']}, "
                    "already in the frozen dataset; the next complete hour opens at 4:00 PM CST"
                )
        result["requests"] = c.stats
    except Exception as e:
        result["note"] = f"probe failed: {type(e).__name__}: {e}"
    return result


def run():
    ROOT.mkdir(parents=True, exist_ok=True)
    batch = H.next_batch_number()
    t0 = time.time()
    print(f"batch {batch}: loading data", flush=True)
    D = H.prep(DATA_DIR)
    T, N = D["T"], D["N"]
    split = int(D["split"])
    times = D["times"]
    hold_lo = FROZEN_HOLDOUT_LO
    if hold_lo >= T:
        raise RuntimeError("frozen holdout start is past the end of the data")
    print(
        f"data {T} hours x {N} tokens; IS 0-{split}; "
        f"purged OOS {split}-{hold_lo}; frozen holdout {hold_lo}-{min(T, FROZEN_T)} "
        f"({H7._chi(times[hold_lo])} -> {H7._chi(times[min(T, FROZEN_T) - 1])})",
        flush=True,
    )

    print("probing GeckoTerminal for a newer complete hourly bar", flush=True)
    probe = probe_new_hourly_bars()
    print(f"  {probe['note']} (new_complete_bars={probe['new_complete_bars']})", flush=True)

    cache = {
        "av24": H.avg_vol(D["V"], 24),
        "hod": ((D["times"].astype(np.int64) % 86400) // 3600).astype(np.int64),
    }
    ncfg = len(entries()) * len(EXITS) * len(SIZING) * len(BALANCES)
    print(
        f"{len(entries())} entries x {len(EXITS)} exits x {len(SIZING)} sizing x {len(BALANCES)} balances = {ncfg}",
        flush=True,
    )
    if ncfg != BATCH_SIZE:
        raise RuntimeError(f"grid is {ncfg}, want {BATCH_SIZE}")
    # 100% fresh families, 0 Double-N
    fams = {f for f, _ in entries()}
    if "washout_gate" in fams or "double_n" in fams:
        raise RuntimeError("Double-N / washout leaked into the fresh grid")
    if len(fams) < 6:
        raise RuntimeError("not enough distinct fresh families")

    print("smoke-testing signals", flush=True)
    problems, counts = smoke_signals(D, cache)
    if problems:
        raise RuntimeError("signal smoke failed: " + "; ".join(problems[:12]))
    print("signal counts by family:", counts, flush=True)

    combos, parts = [], []
    n_sig = 0
    for fam, p in entries():
        sig = make_signal(fam, p, D, cache)
        n_sig += 1
        for e in EXITS:
            tr = E.gen_trades(
                sig, np.asarray(e, float), D["O"], D["H"], D["L"], D["C"], D["last"], D["supply"], False
            )
            combos.append((fam, p, e))
            parts.append(tr)
        if n_sig % 20 == 0:
            print(f"  signals {n_sig}/{len(entries())}  trade-lists {len(parts)}", flush=True)

    lens = np.array([len(t[0]) for t in parts])
    offs = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    tok, ti, to, pin, pout, rs = [np.concatenate([t[k] for t in parts]) for k in range(6)]
    tok, ti, to = tok.astype(np.int64), ti.astype(np.int64), to.astype(np.int64)
    vol_entry = D["V"][np.maximum(ti - 1, 0), tok]
    print(
        f"trade lists {len(combos)}  raw trades {int(lens.sum())}  median trades/list {int(np.median(lens))}",
        flush=True,
    )

    nc, ns, nb = len(combos), len(SIZING), len(BALANCES)
    cfg_combo = np.repeat(np.arange(nc), ns * nb).astype(np.int64)
    cfg_size = np.tile(np.repeat(np.arange(ns), nb), nc)
    cfg_bal = np.tile(BALANCES, nc * ns)
    cfg_frac = np.array([SIZING[s][0] for s in range(ns)])[cfg_size]
    cfg_maxpos = np.array([SIZING[s][1] for s in range(ns)], dtype=np.int64)[cfg_size]

    args = (K.PCT_COST, K.FIXED_SOL, K.RENT_SOL, K.RENT_UNRECOVERED, K.IMPACT_K, K.FILL_CAP)
    res = {}
    hold_hi = min(T, FROZEN_T)
    windows = (
        ("is", 0, split),
        ("oos", split, hold_lo),
        ("full", 0, hold_lo),
        ("hold", hold_lo, hold_hi),
        ("full_all", 0, hold_hi),
    )
    for name, lo, hi in windows:
        t1 = time.time()
        res[name] = E.simulate_many(
            tok, ti, to, pin, pout, D["venue"], vol_entry, D["solpx"], offs,
            cfg_combo, cfg_bal, cfg_frac, cfg_maxpos, lo, hi, *args,
        )
        print(f"  {name} [{lo},{hi}) simulated in {time.time() - t1:.1f}s", flush=True)

    rows = []
    for i in range(BATCH_SIZE):
        c = int(cfg_combo[i])
        fam, p, e = combos[c]
        rows.append(
            row_from(
                i, batch, fam, p, e, float(cfg_frac[i]), int(cfg_maxpos[i]),
                float(cfg_bal[i]), res["oos"][i], res["is"][i], res["full"][i],
                res["hold"][i], res["full_all"][i],
            )
        )

    H7.write_extended_csv(ROOT / f"batch_{batch}.csv", rows)
    H.update_leaderboard(H7.standard_rows(rows))

    oos_pos = [r for r in rows if float(r["oos_profit"]) > 0]
    both = [r for r in rows if r["both_windows"] == "true"]
    flukes = [r for r in oos_pos if r["both_windows"] != "true"]
    both_solid = [r for r in both if int(r["oos_trades"]) >= 8 and int(r["is_trades"]) >= 4]
    hold_pos = [r for r in rows if float(r["holdout_profit"]) > 0]
    both_and_hold = [r for r in both if float(r["holdout_profit"]) > 0]
    oos_pos.sort(key=lambda r: (float(r["oos_profit"]), float(r["oos_ret"])), reverse=True)
    both.sort(key=lambda r: (float(r["oos_profit"]), float(r["is_profit"])), reverse=True)
    both_solid.sort(key=lambda r: (float(r["oos_profit"]), float(r["is_profit"])), reverse=True)
    both_and_hold.sort(key=lambda r: (float(r["holdout_profit"]), float(r["oos_profit"])), reverse=True)
    flukes.sort(key=lambda r: float(r["oos_profit"]), reverse=True)

    by_fam = {}
    for r in rows:
        by_fam.setdefault(r["family"], {"n": 0, "oos_pos": 0, "both": 0, "hold_pos": 0, "both_and_hold": 0})
        by_fam[r["family"]]["n"] += 1
        if float(r["oos_profit"]) > 0:
            by_fam[r["family"]]["oos_pos"] += 1
        if r["both_windows"] == "true":
            by_fam[r["family"]]["both"] += 1
            if float(r["holdout_profit"]) > 0:
                by_fam[r["family"]]["both_and_hold"] += 1
        if float(r["holdout_profit"]) > 0:
            by_fam[r["family"]]["hold_pos"] += 1

    top20 = both[:20]
    print(f"walk-forward batch-{batch} top 20 both-window on pre-holdout (end_bar={hold_lo})", flush=True)
    folds, summaries = WF.walk_forward_shortlist(D, top20, original_split=split, end_bar=hold_lo)
    ranked = sorted(summaries, key=WF.robustness_key, reverse=True)
    WF.write_fold_csv(ROOT / "walkforward_batch8_folds.csv", folds)
    slim = [{k: v for k, v in r.items() if k != "_summary"} for r in ranked]
    WF.write_summary_csv(ROOT / "walkforward_batch8.csv", slim)

    # Frozen holdout eval for batch 8 top 20 (already in rows) + reminder of robust top 10
    print("frozen-holdout evaluation of robust top 10 + batch-8 top 20", flush=True)
    caches = {}
    hold_rows = []
    robust = H.load_csv(ROOT / "robust_leaderboard.csv")[:10]
    for r in robust:
        is_m, purged, hold, pre, _ = H7.eval_row_windows(r, D, caches, split, hold_lo, hold_hi)
        rec = H7.holdout_record(
            r, hold, purged, is_m, pre, hold_lo, hold_hi, times,
            "includes_holdout (selected in batches 1-7)",
        )
        rec["orig_oos_profit"] = r.get("orig_oos_profit", r.get("oos_profit", ""))
        rec["orig_is_ret"] = r.get("orig_is_ret", r.get("is_ret", ""))
        hold_rows.append(rec)
        print(
            f"  {r['id']} hold_ret={hold['ret']} hold_profit={hold['profit']} tr={hold['trades']}",
            flush=True,
        )
    by_id = {r["id"]: r for r in rows}
    for r in top20:
        src = by_id[r["id"]]
        rec = {
            "id": src["id"],
            "batch": src["batch"],
            "family": src["family"],
            "entry": src["entry"],
            "exit": src["exit"],
            "sizing": src["sizing"],
            "balance": src["balance"],
            "selected_on": "pre_holdout_only",
            "hold_start": H7._chi(times[hold_lo]),
            "hold_end": H7._chi(times[hold_hi - 1]),
            "hold_bars": int(hold_hi - hold_lo),
            "hold_profit": src["holdout_profit"],
            "hold_ret": src["holdout_ret"],
            "hold_trades": src["holdout_trades"],
            "hold_win": src["holdout_win"],
            "hold_dd": src["holdout_dd"],
            "hold_sharpe": src["holdout_sharpe"],
            "hold_pf": src["holdout_pf"],
            "hold_profitable": "true" if float(src["holdout_profit"]) > 0 else "false",
            "purged_oos_profit": src["oos_profit"],
            "purged_oos_ret": src["oos_ret"],
            "purged_oos_trades": src["oos_trades"],
            "is_ret": src["is_ret"],
            "is_trades": src["is_trades"],
            "full_pre_ret": src["full_ret"],
            "orig_oos_profit": src["oos_profit"],
            "orig_is_ret": src["is_ret"],
            "params": src["params"],
            "exit_params": src["exit_params"],
        }
        hold_rows.append(rec)
    H7.write_hold_csv(ROOT / "holdout_eval_batch8.csv", hold_rows)

    old = H.load_csv(ROOT / "robust_leaderboard.csv") if (ROOT / "robust_leaderboard.csv").exists() else []
    hold_by = {r["id"]: r for r in hold_rows if str(r["id"]).startswith(f"H{batch}-")}
    new_pass = []
    for s in slim:
        h = hold_by.get(s["id"])
        if s.get("passed") == "true" and h and h["hold_profitable"] == "true":
            new_pass.append(s)
    merged = {r["id"]: r for r in old}
    for r in new_pass:
        merged[r["id"]] = {k: r[k] for k in WF.ROBUST_FIELDS if k in r}
    passed = [r for r in merged.values() if str(r.get("passed", "")).lower() == "true"]
    passed.sort(key=WF.robustness_key, reverse=True)
    WF.write_summary_csv(ROOT / "robust_leaderboard.csv", passed, fields=WF.ROBUST_FIELDS)

    top10_hold = [r for r in hold_rows if str(r.get("selected_on", "")).startswith("includes_holdout")]
    b8_hold = [r for r in hold_rows if r.get("selected_on") == "pre_holdout_only"]

    payload = {
        "probe": probe,
        "unseen_new_bars": None,
        "frozen_holdout": {
            "lo": hold_lo,
            "hi": hold_hi,
            "start": H7._chi(times[hold_lo]),
            "end": H7._chi(times[hold_hi - 1]),
            "bars": int(hold_hi - hold_lo),
        },
        "robust_top10_frozen_holdout": {
            "n_profitable": sum(1 for r in top10_hold if r["hold_profitable"] == "true"),
            "rows": top10_hold,
        },
        "batch8_top20_frozen_holdout": {
            "n_profitable": sum(1 for r in b8_hold if r["hold_profitable"] == "true"),
            "rows": b8_hold,
        },
        "n_robust_board": len(passed),
        "n_batch8_added_to_robust": len(new_pass),
    }
    with open(ROOT / "holdout_eval_batch8.json", "w") as f:
        json.dump(payload, f, indent=2)

    summary = {
        "batch": batch,
        "n": len(rows),
        "seconds": round(time.time() - t0, 1),
        "fresh_family_share": 1.0,
        "families": sorted(fams),
        "data": {
            "hours": int(T),
            "tokens": int(N),
            "split_bar": int(split),
            "holdout_lo": int(hold_lo),
            "ranking": "IS + purged OOS (frozen holdout after 3961 excluded)",
        },
        "oos_profitable": len(oos_pos),
        "both_windows_profitable": len(both),
        "both_windows_solid": len(both_solid),
        "oos_only_flukes": len(flukes),
        "holdout_profitable": len(hold_pos),
        "both_and_holdout": len(both_and_hold),
        "by_family": by_fam,
        "top_both_windows": H7.pack(both, 12, 0, holdout=True),
        "top_both_windows_solid": H7.pack(both_solid, 12, 8, holdout=True),
        "top_both_and_holdout": H7.pack(both_and_hold, 12, 0, holdout=True),
        "top_flukes": H7.pack(flukes, 8, 0),
        "walkforward_pre_holdout_top20": [
            {
                "id": r["id"],
                "family": r.get("family", ""),
                "passed": r["passed"],
                "wf_roll_oos_profitable_pct": r["wf_roll_oos_profitable_pct"],
                "wf_roll_median_oos_ret": r["wf_roll_median_oos_ret"],
                "wf_full_ret": r["wf_full_ret"],
            }
            for r in ranked
        ],
        "holdout": payload,
    }
    with open(ROOT / f"batch_{batch}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"batch {batch} done: {len(rows)} variants, {len(oos_pos)} purged-OOS profitable, "
        f"{len(both)} both-windows, {len(both_and_hold)} both+holdout, "
        f"robust top10 frozen-holdout profitable {payload['robust_top10_frozen_holdout']['n_profitable']}/10, "
        f"{time.time()-t0:.1f}s",
        flush=True,
    )
    return summary


if __name__ == "__main__":
    run()
