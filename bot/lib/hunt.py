#!/usr/bin/env python3
"""Strategy hunter batch runner.

Discovers and backtests NEW families that are not in the main solstrats grid.
Vendored copy of the strategy-hunter signal library.
Reuses solstrats data loading, cost model, trade generation, and portfolio sim.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit, prange

ROOT = Path(__file__).resolve().parent
DATA_DIR = os.environ.get("SOLSTRATS_DATA", str(Path(__file__).resolve().parent.parent / "run" / "data"))
MAIN = str(Path(__file__).resolve().parent)
sys.path.insert(0, MAIN)

from solstrats import costs as K  # noqa: E402
from solstrats import engine as E  # noqa: E402
from solstrats.describe import exit_text, sizing_text  # noqa: E402
from run_backtest import prep  # noqa: E402

BATCH_SIZE = 10_000
NONE = E.NONE

# ---------------------------------------------------------------------------
# Research-backed families that are NOT in the main grid
# (breakout, ma_cross, rsi_revert, vol_spike, dip_buy, moonshot, desk,
#  fast_mc, graduation, bonding_curve, narrative, flash_wick).
# ---------------------------------------------------------------------------

FAMILIES = [
    "vwap_reclaim",
    "rel_strength",
    "pullback_ma",
    "failed_breakdown",
    "theme_laggard",
    "post_run_pullback",
    "climax_reversal",
    "sol_risk_on",
    "dryup_expansion",
    "session_orb",
]


def entry_text(fam: str, p: tuple) -> str:
    def pct(x):
        return f"{x * 100:g}%"

    if fam == "vwap_reclaim":
        s = f"Buy when price reclaims its {int(p[0])}h VWAP after trading below it"
        return s + (f" on {p[1]:g}x average volume" if p[1] else "")
    if fam == "rel_strength":
        s = f"Buy when the token outperforms SOL by {pct(p[1])} over {int(p[0])}h"
        if p[2]:
            s += " after a pullback from its 24h high of at least 5%"
        return s
    if fam == "pullback_ma":
        s = (
            f"Buy a bounce back above the {int(p[0])}h average while price stays "
            f"above the {int(p[1])}h trend average"
        )
        return s + (f" on {p[2]:g}x average volume" if p[2] else "")
    if fam == "failed_breakdown":
        s = (
            f"Buy a same-bar spring: the hour breaks the prior {int(p[0])}h low "
            "but closes back above it"
        )
        return s + (f" on {p[1]:g}x average volume" if p[1] else "")
    if fam == "theme_laggard":
        return (
            f"Daily, buy laggards in the hottest theme (by {int(p[0])}h return) "
            f"that trail the theme by at least {pct(p[1])}"
            + (f", only if the theme itself is up {pct(p[2])}+" if p[2] else "")
        )
    if fam == "post_run_pullback":
        return (
            f"Buy the first bounce after a {pct(p[0])}+ run over {int(p[1])}h "
            f"that has pulled back {pct(p[2])} from the run high"
        )
    if fam == "climax_reversal":
        return (
            f"Buy the first green hour after a climax dump of {pct(p[0])}+ "
            f"on {p[1]:g}x average volume (confirmation bar, not the dump bar)"
        )
    if fam == "sol_risk_on":
        s = (
            f"Buy a green hour when SOL has dropped {pct(p[0])}+ over {int(p[1])}h"
        )
        if p[2]:
            s += " and the token has held up better than SOL"
        return s
    if fam == "dryup_expansion":
        s = (
            f"Buy a green expansion hour of {p[1]:g}x volume after {int(p[0])}h "
            "of below-average volume"
        )
        if p[2]:
            s += f" with the hour up {pct(p[2])}+"
        return s
    if fam == "session_orb":
        s = f"Buy a close above the first-{int(p[0])}h UTC opening-range high"
        if p[1]:
            s += f" on {p[1]:g}x average volume"
        if p[2]:
            s += ", established tokens only"
        return s
    return fam


def entries():
    """Exactly 80 entry variants (10 families x 8 param sets)."""
    P = lambda *a: tuple(list(a) + [0.0] * (4 - len(a)))
    Elist = []
    for L, vm in ((12, 0), (12, 2), (24, 0), (24, 2), (48, 0), (48, 2), (72, 0), (72, 2)):
        Elist.append(("vwap_reclaim", P(L, vm)))
    for L, thr, fade in (
        (24, 0.05, 0),
        (24, 0.05, 1),
        (24, 0.15, 0),
        (24, 0.15, 1),
        (72, 0.05, 0),
        (72, 0.05, 1),
        (72, 0.15, 0),
        (72, 0.15, 1),
    ):
        Elist.append(("rel_strength", P(L, thr, fade)))
    for fa, sl, vm in (
        (8, 72, 0),
        (8, 72, 2),
        (8, 168, 0),
        (8, 168, 2),
        (24, 72, 0),
        (24, 72, 2),
        (24, 168, 0),
        (24, 168, 2),
    ):
        Elist.append(("pullback_ma", P(fa, sl, vm)))
    for L, vm in ((12, 0), (12, 2), (24, 0), (24, 2), (48, 0), (48, 2), (72, 0), (72, 2)):
        Elist.append(("failed_breakdown", P(L, vm)))
    for L, lag, mtr in (
        (24, 0.0, 0.0),
        (24, 0.10, 0.0),
        (24, 0.0, 0.05),
        (24, 0.10, 0.05),
        (72, 0.0, 0.0),
        (72, 0.10, 0.0),
        (72, 0.0, 0.05),
        (72, 0.10, 0.05),
    ):
        Elist.append(("theme_laggard", P(L, lag, mtr)))
    for run, w, dip in (
        (0.4, 24, 0.10),
        (0.4, 24, 0.20),
        (0.4, 72, 0.10),
        (0.4, 72, 0.20),
        (1.0, 24, 0.10),
        (1.0, 24, 0.20),
        (1.0, 72, 0.10),
        (1.0, 72, 0.20),
    ):
        Elist.append(("post_run_pullback", P(run, w, dip)))
    for d, vm in ((0.08, 2), (0.08, 4), (0.15, 2), (0.15, 4), (0.25, 2), (0.25, 4), (0.40, 2), (0.40, 4)):
        Elist.append(("climax_reversal", P(d, vm)))
    for dmp, w, hold in (
        (0.03, 6, 0),
        (0.03, 6, 1),
        (0.03, 24, 0),
        (0.03, 24, 1),
        (0.06, 6, 0),
        (0.06, 6, 1),
        (0.06, 24, 0),
        (0.06, 24, 1),
    ):
        Elist.append(("sol_risk_on", P(dmp, w, hold)))
    for w, vm, mr in (
        (6, 2, 0.0),
        (6, 2, 0.02),
        (6, 4, 0.0),
        (6, 4, 0.02),
        (12, 2, 0.0),
        (12, 2, 0.02),
        (12, 4, 0.0),
        (12, 4, 0.02),
    ):
        Elist.append(("dryup_expansion", P(w, vm, mr)))
    for k, vm, est in (
        (2, 0, 0),
        (2, 0, 1),
        (2, 2, 0),
        (2, 2, 1),
        (4, 0, 0),
        (4, 0, 1),
        (4, 2, 0),
        (4, 2, 1),
    ):
        Elist.append(("session_orb", P(k, vm, est)))
    if len(Elist) != 80:
        raise RuntimeError(f"expected 80 entries, got {len(Elist)}")
    return Elist


# 5 exits x 5 sizing x 3 balances x 80 entries = 6,000... need 10,000
# 5 exits x 5 sizing x 5 balances x 80 entries = 10,000 exactly.
EXITS = [
    (0.05, 0.30, NONE, 0.0, 24, NONE, 1.0),  # scalp
    (0.50, 0.20, NONE, 0.0, 168, NONE, 1.0),  # standard swing
    (1.00, 0.35, NONE, 0.0, 240, NONE, 1.0),  # wider target
    (NONE, 0.35, 0.50, 0.25, 720, NONE, 1.0),  # uncapped trail
    (NONE, 0.15, 0.25, 0.15, 240, NONE, 1.0),  # tight trail
]
SIZING = [(0.08, 12), (0.20, 5), (0.50, 2), (1.00, 1), (0.05, 20)]
BALANCES = np.array([5.0, 25.0, 100.0, 250.0, 280.0], dtype=np.float64)


# ---------------------------------------------------------------------------
# Precompute helpers (numpy; no look-ahead)
# ---------------------------------------------------------------------------

def _rolling_sum(a, L):
    cs = np.cumsum(a, axis=0)
    out = cs.copy()
    out[L:] = cs[L:] - cs[:-L]
    out[: L - 1] = np.nan
    return out


def rolling_vwap(C, V, L):
    pv = np.nan_to_num(C, nan=0.0) * V
    s_pv = _rolling_sum(pv, L)
    s_v = _rolling_sum(V, L)
    with np.errstate(invalid="ignore", divide="ignore"):
        vw = s_pv / s_v
    vw[s_v <= 0] = np.nan
    return vw


def rolling_mean(C, L):
    x = np.nan_to_num(C, nan=0.0)
    s = _rolling_sum(x, L)
    return s / L


def avg_vol(V, L=24):
    return _rolling_sum(V, L) / L


def opening_range_high(H, first, times, K):
    T, N = H.shape
    day = (times.astype(np.int64) // 86400)
    orh = np.full((T, N), np.nan)
    for j in range(N):
        f0 = int(first[j])
        if f0 < 0:
            continue
        t = f0
        while t < T:
            d = day[t]
            mx = -1e99
            t1 = t
            cnt = 0
            while t1 < T and day[t1] == d and cnt < K:
                h = H[t1, j]
                if h > mx:
                    mx = h
                t1 += 1
                cnt += 1
            if cnt < K or mx <= 0:
                # range not formed; skip the rest of the day
                while t1 < T and day[t1] == d:
                    t1 += 1
                t = t1 if t1 > t else t + 1
                continue
            t2 = t1
            while t2 < T and day[t2] == d:
                orh[t2, j] = mx
                t2 += 1
            t = t2 if t2 > t else t + 1
    return orh


def theme_stats(C, theme, grp, L):
    """theme_ret[t, th] = mean L-hour return of theme th at bar t; hot[t] = best theme."""
    T, N = C.shape
    ntheme = int(theme.max()) + 1
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.full((T, N), np.nan)
        r[L:] = C[L:] / C[:-L] - 1.0
    theme_ret = np.full((T, ntheme), np.nan)
    hot = np.full(T, -1, dtype=np.int64)
    other = ntheme - 1
    for t in range(L, T):
        best, bi = -1e9, -1
        for th in range(other):  # "other" is not a tradable narrative
            sel = (theme == th) & ~np.isnan(r[t]) & (grp != 2)
            if sel.sum() >= 2:
                v = float(np.nanmean(r[t, sel]))
                theme_ret[t, th] = v
                if v > best:
                    best, bi = v, th
        hot[t] = bi
    return r, theme_ret, hot


# ---------------------------------------------------------------------------
# Numba signal kernels. Signal at close of bar t; fill at open of t+1.
# ---------------------------------------------------------------------------

@njit(cache=True, parallel=True)
def sig_vwap_reclaim(C, V, first, grp, vwap, av24, L, vm, t_hi):
    T, N = C.shape
    sig = np.zeros((T, N), dtype=np.bool_)
    for j in prange(N):
        if grp[j] == 2 or first[j] < 0:
            continue
        f0 = first[j]
        start = max(f0 + L + 1, L)
        for t in range(start, t_hi):
            vw = vwap[t, j]
            prev = vwap[t - 1, j]
            c = C[t, j]
            p = C[t - 1, j]
            if not (c > 0 and p > 0 and vw == vw and prev == prev):
                continue
            if p >= prev:
                continue
            if c < vw:
                continue
            if vm > 0:
                a = av24[t, j]
                if not (a == a) or a <= 0 or V[t, j] < vm * a:
                    continue
            sig[t, j] = True
    return sig


@njit(cache=True, parallel=True)
def sig_rel_strength(C, sol, first, grp, L, thr, fade, t_hi):
    T, N = C.shape
    sig = np.zeros((T, N), dtype=np.bool_)
    for j in prange(N):
        if grp[j] == 2 or first[j] < 0:
            continue
        f0 = first[j]
        start = max(f0 + L, L, 24 if fade > 0 else L)
        for t in range(start, t_hi):
            c = C[t, j]
            b = C[t - L, j]
            s1, s0 = sol[t], sol[t - L]
            if not (c > 0 and b > 0 and s1 > 0 and s0 > 0):
                continue
            rs = (c / b - 1.0) - (s1 / s0 - 1.0)
            if rs < thr:
                continue
            if fade > 0:
                mx = 0.0
                for k in range(t - 23, t + 1):
                    if C[k, j] > mx:
                        mx = C[k, j]
                if mx <= 0 or c > mx * 0.95:
                    continue
            sig[t, j] = True
    return sig


@njit(cache=True, parallel=True)
def sig_pullback_ma(C, V, first, grp, ma_f, ma_s, av24, vm, sl, t_hi):
    T, N = C.shape
    sig = np.zeros((T, N), dtype=np.bool_)
    for j in prange(N):
        if grp[j] == 2 or first[j] < 0:
            continue
        f0 = first[j]
        start = max(f0 + sl + 1, sl + 1)
        for t in range(start, t_hi):
            c = C[t, j]
            p = C[t - 1, j]
            fs, ss = ma_f[t, j], ma_s[t, j]
            if not (c > 0 and p > 0 and fs == fs and ss == ss):
                continue
            if c <= ss:
                continue
            if p > fs:
                continue
            if c <= fs:
                continue
            if vm > 0:
                a = av24[t, j]
                if not (a == a) or a <= 0 or V[t, j] < vm * a:
                    continue
            sig[t, j] = True
    return sig


@njit(cache=True, parallel=True)
def sig_failed_breakdown(L_, H, C, V, first, grp, av24, Lb, vm, t_hi):
    T, N = C.shape
    sig = np.zeros((T, N), dtype=np.bool_)
    for j in prange(N):
        if grp[j] == 2 or first[j] < 0:
            continue
        f0 = first[j]
        start = max(f0 + Lb + 1, Lb)
        for t in range(start, t_hi):
            c = C[t, j]
            if not (c > 0):
                continue
            supp = 1e300
            for k in range(t - Lb, t):
                lv = L_[k, j]
                if lv < supp:
                    supp = lv
            if not (supp > 0 and supp < 1e299):
                continue
            if L_[t, j] >= supp:
                continue
            if c <= supp:
                continue
            if vm > 0:
                a = av24[t, j]
                if not (a == a) or a <= 0 or V[t, j] < vm * a:
                    continue
            sig[t, j] = True
    return sig


@njit(cache=True, parallel=True)
def sig_theme_laggard(C, first, grp, theme, hod, tok_ret, theme_ret, hot, L, lag, mtr, t_hi):
    T, N = C.shape
    sig = np.zeros((T, N), dtype=np.bool_)
    ntheme = theme_ret.shape[1]
    for j in prange(N):
        if grp[j] == 2 or first[j] < 0:
            continue
        th = theme[j]
        if th < 0 or th >= ntheme - 1:
            continue
        f0 = first[j]
        start = max(f0 + L, L)
        for t in range(start, t_hi):
            if hod[t] != 0:
                continue
            if hot[t] != th:
                continue
            tr = theme_ret[t, th]
            rr = tok_ret[t, j]
            if not (tr == tr and rr == rr):
                continue
            if tr < mtr:
                continue
            if rr > tr - lag:
                continue
            if C[t, j] <= 0:
                continue
            sig[t, j] = True
    return sig


@njit(cache=True, parallel=True)
def sig_post_run_pullback(H, C, first, grp, run, W, dip, t_hi):
    T, N = C.shape
    sig = np.zeros((T, N), dtype=np.bool_)
    for j in prange(N):
        if grp[j] == 2 or first[j] < 0:
            continue
        f0 = first[j]
        start = max(f0 + W + 1, W)
        for t in range(start, t_hi):
            c = C[t, j]
            p = C[t - 1, j]
            b = C[t - W, j]
            if not (c > 0 and p > 0 and b > 0):
                continue
            mx = 0.0
            for k in range(t - W, t + 1):
                if H[k, j] > mx:
                    mx = H[k, j]
            if mx <= 0 or mx / b - 1.0 < run:
                continue
            if c > mx * (1.0 - dip):
                continue
            if c <= p:  # require a bounce close
                continue
            sig[t, j] = True
    return sig


@njit(cache=True, parallel=True)
def sig_climax_reversal(C, V, first, grp, av24, dmp, vm, t_hi):
    T, N = C.shape
    sig = np.zeros((T, N), dtype=np.bool_)
    for j in prange(N):
        if grp[j] == 2 or first[j] < 0:
            continue
        f0 = first[j]
        start = max(f0 + 26, 26)
        for t in range(start, t_hi):
            c = C[t, j]
            p = C[t - 1, j]
            pp = C[t - 2, j]
            if not (c > 0 and p > 0 and pp > 0):
                continue
            if p / pp - 1.0 > -dmp:
                continue
            a = av24[t - 1, j]
            if not (a == a) or a <= 0 or V[t - 1, j] < vm * a:
                continue
            if c <= p:  # confirmation must be green vs prior close
                continue
            sig[t, j] = True
    return sig


@njit(cache=True, parallel=True)
def sig_sol_risk_on(C, sol, first, grp, dmp, W, hold, t_hi):
    T, N = C.shape
    sig = np.zeros((T, N), dtype=np.bool_)
    for j in prange(N):
        if grp[j] == 2 or first[j] < 0:
            continue
        f0 = first[j]
        start = max(f0 + W + 1, W)
        for t in range(start, t_hi):
            s1, s0 = sol[t], sol[t - W]
            if not (s1 > 0 and s0 > 0):
                continue
            sr = s1 / s0 - 1.0
            if sr > -dmp:
                continue
            c = C[t, j]
            p = C[t - 1, j]
            b = C[t - W, j]
            if not (c > 0 and p > 0 and b > 0):
                continue
            if c <= p:
                continue
            if hold > 0 and (c / b - 1.0) <= sr:
                continue
            sig[t, j] = True
    return sig


@njit(cache=True, parallel=True)
def sig_dryup_expansion(C, V, first, grp, av24, W, vm, mr, t_hi):
    T, N = C.shape
    sig = np.zeros((T, N), dtype=np.bool_)
    for j in prange(N):
        if grp[j] == 2 or first[j] < 0:
            continue
        f0 = first[j]
        start = max(f0 + 24 + W, 24 + W)
        for t in range(start, t_hi):
            a = av24[t, j]
            if not (a == a) or a <= 0:
                continue
            dry = True
            for k in range(t - W, t):
                if V[k, j] >= a:
                    dry = False
                    break
            if not dry:
                continue
            if V[t, j] < vm * a:
                continue
            c = C[t, j]
            p = C[t - 1, j]
            if not (c > 0 and p > 0):
                continue
            if c / p - 1.0 < mr:
                continue
            if c <= p:
                continue
            sig[t, j] = True
    return sig


@njit(cache=True, parallel=True)
def sig_session_orb(C, V, first, grp, orh, av24, vm, est_only, t_hi):
    T, N = C.shape
    sig = np.zeros((T, N), dtype=np.bool_)
    for j in prange(N):
        if grp[j] == 2 or first[j] < 0:
            continue
        if est_only > 0 and grp[j] != 0:
            continue
        f0 = first[j]
        start = max(f0 + 5, 5)
        for t in range(start, t_hi):
            oh = orh[t, j]
            c = C[t, j]
            p = C[t - 1, j]
            if not (c > 0 and oh == oh and oh > 0):
                continue
            if c <= oh:
                continue
            # first close of the day above the OR high (avoid repeating every hour)
            if p > 0 and orh[t - 1, j] == orh[t - 1, j] and p > orh[t - 1, j]:
                continue
            if vm > 0:
                a = av24[t, j]
                if not (a == a) or a <= 0 or V[t, j] < vm * a:
                    continue
            sig[t, j] = True
    return sig


def make_signal(fam, p, D, cache):
    C, V, H, L = D["C"], D["V"], D["H"], D["L"]
    first, grp = D["first"], D["grp"]
    t_hi = D["T"] - 1
    av24 = cache["av24"]
    if fam == "vwap_reclaim":
        Lbar, vm = int(p[0]), float(p[1])
        key = ("vwap", Lbar)
        if key not in cache:
            cache[key] = rolling_vwap(C, V, Lbar)
        return sig_vwap_reclaim(C, V, first, grp, cache[key], av24, Lbar, vm, t_hi)
    if fam == "rel_strength":
        return sig_rel_strength(C, D["solpx"], first, grp, int(p[0]), float(p[1]), int(p[2]), t_hi)
    if fam == "pullback_ma":
        fa, sl, vm = int(p[0]), int(p[1]), float(p[2])
        kf, ks = ("ma", fa), ("ma", sl)
        if kf not in cache:
            cache[kf] = rolling_mean(C, fa)
        if ks not in cache:
            cache[ks] = rolling_mean(C, sl)
        return sig_pullback_ma(C, V, first, grp, cache[kf], cache[ks], av24, vm, sl, t_hi)
    if fam == "failed_breakdown":
        return sig_failed_breakdown(L, H, C, V, first, grp, av24, int(p[0]), float(p[1]), t_hi)
    if fam == "theme_laggard":
        Lbar, lag, mtr = int(p[0]), float(p[1]), float(p[2])
        key = ("theme", Lbar)
        if key not in cache:
            cache[key] = theme_stats(C, D["theme"], grp, Lbar)
        tok_ret, theme_ret, hot = cache[key]
        return sig_theme_laggard(
            C, first, grp, D["theme"], cache["hod"], tok_ret, theme_ret, hot, Lbar, lag, mtr, t_hi
        )
    if fam == "post_run_pullback":
        return sig_post_run_pullback(H, C, first, grp, float(p[0]), int(p[1]), float(p[2]), t_hi)
    if fam == "climax_reversal":
        return sig_climax_reversal(C, V, first, grp, av24, float(p[0]), float(p[1]), t_hi)
    if fam == "sol_risk_on":
        return sig_sol_risk_on(C, D["solpx"], first, grp, float(p[0]), int(p[1]), int(p[2]), t_hi)
    if fam == "dryup_expansion":
        return sig_dryup_expansion(C, V, first, grp, av24, int(p[0]), float(p[1]), float(p[2]), t_hi)
    if fam == "session_orb":
        Kbar, vm, est = int(p[0]), float(p[1]), int(p[2])
        key = ("orb", Kbar)
        if key not in cache:
            cache[key] = opening_range_high(H, first, D["times"], Kbar)
        return sig_session_orb(C, V, first, grp, cache[key], av24, vm, est, t_hi)
    raise KeyError(fam)


# ---------------------------------------------------------------------------
# Batch I/O
# ---------------------------------------------------------------------------

FIELDS = [
    "id",
    "batch",
    "family",
    "entry",
    "exit",
    "sizing",
    "balance",
    "oos_end",
    "oos_profit",
    "oos_ret",
    "oos_win",
    "oos_trades",
    "oos_dd",
    "oos_sharpe",
    "oos_pf",
    "oos_avg_win",
    "oos_avg_loss",
    "is_end",
    "is_profit",
    "is_ret",
    "is_sharpe",
    "is_trades",
    "is_win",
    "is_dd",
    "full_end",
    "full_ret",
    "full_win",
    "full_trades",
    "full_dd",
    "full_fees",
    "both_windows",
    "params",
    "exit_params",
]


def next_batch_number():
    existing = sorted(glob.glob(str(ROOT / "batch_*.csv")))
    if not existing:
        return 1
    nums = []
    for p in existing:
        stem = Path(p).stem  # batch_1
        try:
            nums.append(int(stem.split("_", 1)[1]))
        except ValueError:
            pass
    return (max(nums) if nums else 0) + 1


def row_from(i, batch, fam, p, e, frac, maxpos, bal, is_, oos, full):
    return {
        "id": f"H{batch}-{i:05d}",
        "batch": batch,
        "family": fam,
        "entry": entry_text(fam, p),
        "exit": exit_text(e),
        "sizing": sizing_text(frac, maxpos),
        "balance": float(bal),
        "oos_end": round(float(oos[0]), 4),
        "oos_profit": round(float(oos[0] - bal), 4),
        "oos_ret": round(float(oos[1]), 4),
        "oos_win": round(float(oos[13]), 4),
        "oos_trades": int(oos[4]),
        "oos_dd": round(float(oos[2]), 4),
        "oos_sharpe": round(float(oos[3]), 4),
        "oos_pf": round(float(oos[12]), 4),
        "oos_avg_win": round(float(oos[14]), 4),
        "oos_avg_loss": round(float(oos[15]), 4),
        "is_end": round(float(is_[0]), 4),
        "is_profit": round(float(is_[0] - bal), 4),
        "is_ret": round(float(is_[1]), 4),
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
        "both_windows": both_windows_flag(float(oos[0] - bal), float(is_[1])),
        "params": json.dumps(list(map(float, p))),
        "exit_params": json.dumps(list(map(float, e))),
    }


def both_windows_flag(oos_profit, is_ret):
    """Dashboard-friendly boolean: profitable in both IS and OOS windows."""
    return "true" if float(oos_profit) > 0 and float(is_ret) > 0 else "false"


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def current_main_best():
    """Best OOS profit on the main hourly grid (results_smoke)."""
    path = os.path.join(MAIN, "results_smoke", "top100_by_profit.csv")
    if not os.path.exists(path):
        return None
    rows = load_csv(path)
    if not rows:
        return None
    return max(rows, key=lambda r: float(r["oos_profit"]))


def ensure_both_windows(row):
    v = str(row.get("both_windows", "")).strip().lower()
    if v in ("true", "false"):
        row["both_windows"] = v
        return row
    row["both_windows"] = both_windows_flag(row.get("oos_profit", 0), row.get("is_ret", 0))
    return row


def update_leaderboard(new_rows):
    path = ROOT / "leaderboard.csv"
    all_rows = list(new_rows)
    if path.exists():
        all_rows = load_csv(path) + all_rows
    all_rows = [ensure_both_windows(r) for r in all_rows]
    all_rows.sort(key=lambda r: (float(r["oos_profit"]), float(r["oos_ret"])), reverse=True)
    # keep a deep board so later batches can compare
    write_csv(path, all_rows)
    return all_rows


def smoke_signals(D, cache):
    """Sanity: shapes, no trades on SOL, some families fire, no all-True."""
    T, N = D["T"], D["N"]
    problems = []
    counts = {}
    for fam, p in entries():
        sig = make_signal(fam, p, D, cache)
        if sig.shape != (T, N):
            problems.append(f"{fam} bad shape {sig.shape}")
            continue
        if D["sol_idx"] >= 0 and sig[:, D["sol_idx"]].any():
            problems.append(f"{fam} signaled on SOL")
        n = int(sig.sum())
        counts.setdefault(fam, 0)
        counts[fam] += n
        if n == sig.size:
            problems.append(f"{fam} all True")
        # look-ahead-ish: first bar of a token should never signal
        for j in range(N):
            f0 = int(D["first"][j])
            if f0 >= 0 and f0 < T and sig[f0, j]:
                problems.append(f"{fam} signal on first bar token {j}")
                break
    return problems, counts


def run():
    ROOT.mkdir(parents=True, exist_ok=True)
    batch = next_batch_number()
    t0 = time.time()
    print(f"batch {batch}: loading data", flush=True)
    D = prep(DATA_DIR)
    T, N = D["T"], D["N"]
    split = D["split"]
    hod = ((D["times"].astype(np.int64) % 86400) // 3600).astype(np.int64)
    cache = {"av24": avg_vol(D["V"], 24), "hod": hod}
    print(
        f"data {T} hours x {N} tokens; split bar {split}; "
        f"{len(entries())} entries x {len(EXITS)} exits x {len(SIZING)} sizing x {len(BALANCES)} balances "
        f"= {len(entries()) * len(EXITS) * len(SIZING) * len(BALANCES)} configs",
        flush=True,
    )

    print("smoke-testing signals", flush=True)
    problems, counts = smoke_signals(D, cache)
    if problems:
        raise RuntimeError("signal smoke failed: " + "; ".join(problems[:8]))
    print("signal counts by family (sum over 8 param sets):", counts, flush=True)

    # Compile trade gen / sim once
    empty_sig = np.zeros((T, N), dtype=np.bool_)
    E.gen_trades(
        empty_sig,
        np.asarray(EXITS[0], float),
        D["O"],
        D["H"],
        D["L"],
        D["C"],
        D["last"],
        D["supply"],
        False,
    )

    combos = []
    parts = []
    n_sig = 0
    for fam, p in entries():
        sig = make_signal(fam, p, D, cache)
        n_sig += 1
        for e in EXITS:
            tr = E.gen_trades(
                sig,
                np.asarray(e, float),
                D["O"],
                D["H"],
                D["L"],
                D["C"],
                D["last"],
                D["supply"],
                False,
            )
            combos.append((fam, p, e))
            parts.append(tr)
        if n_sig % 10 == 0:
            print(f"  signals {n_sig}/{len(entries())}  trade-lists {len(parts)}", flush=True)

    lens = np.array([len(t[0]) for t in parts])
    offs = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    tok, ti, to, pin, pout, rs = [np.concatenate([t[k] for t in parts]) for k in range(6)]
    tok = tok.astype(np.int64)
    ti = ti.astype(np.int64)
    to = to.astype(np.int64)
    vol_entry = D["V"][np.maximum(ti - 1, 0), tok]
    print(
        f"trade lists {len(combos)}  raw trades {int(lens.sum())}  "
        f"median trades/list {int(np.median(lens))}",
        flush=True,
    )

    nc = len(combos)
    ns, nb = len(SIZING), len(BALANCES)
    cfg_combo = np.repeat(np.arange(nc), ns * nb).astype(np.int64)
    cfg_size = np.tile(np.repeat(np.arange(ns), nb), nc)
    cfg_bal = np.tile(BALANCES, nc * ns)
    cfg_frac = np.array([SIZING[s][0] for s in range(ns)])[cfg_size]
    cfg_maxpos = np.array([SIZING[s][1] for s in range(ns)], dtype=np.int64)[cfg_size]

    # Exactly 10,000 variants
    if cfg_combo.size != BATCH_SIZE:
        raise RuntimeError(f"grid is {cfg_combo.size}, want {BATCH_SIZE}")

    args = (K.PCT_COST, K.FIXED_SOL, K.RENT_SOL, K.RENT_UNRECOVERED, K.IMPACT_K, K.FILL_CAP)
    res = {}
    for name, lo, hi in (("is", 0, split), ("oos", split, T), ("full", 0, T)):
        t1 = time.time()
        res[name] = E.simulate_many(
            tok,
            ti,
            to,
            pin,
            pout,
            D["venue"],
            vol_entry,
            D["solpx"],
            offs,
            cfg_combo,
            cfg_bal,
            cfg_frac,
            cfg_maxpos,
            lo,
            hi,
            *args,
        )
        print(f"sim {name} {res[name].shape} in {time.time() - t1:.1f}s", flush=True)

    rows = []
    for i in range(BATCH_SIZE):
        c = int(cfg_combo[i])
        fam, p, e = combos[c]
        s = int(cfg_size[i])
        frac, maxpos = SIZING[s]
        rows.append(
            row_from(i + 1, batch, fam, p, e, frac, maxpos, float(cfg_bal[i]), res["is"][i], res["oos"][i], res["full"][i])
        )

    out = ROOT / f"batch_{batch}.csv"
    write_csv(out, rows)
    board = update_leaderboard(rows)

    # summary stats (honest)
    oos_tr = [r for r in rows if int(r["oos_trades"]) > 0]
    oos_pos = [r for r in oos_tr if float(r["oos_profit"]) > 0]
    both_pos = [r for r in oos_pos if float(r["is_ret"]) > 0]
    fam_stats = {}
    for r in rows:
        fam_stats.setdefault(r["family"], {"n": 0, "with_trades": 0, "pos": 0, "best": -1e99, "best_row": None})
        s = fam_stats[r["family"]]
        s["n"] += 1
        if int(r["oos_trades"]) > 0:
            s["with_trades"] += 1
        if float(r["oos_profit"]) > 0:
            s["pos"] += 1
        if float(r["oos_profit"]) > s["best"]:
            s["best"] = float(r["oos_profit"])
            s["best_row"] = r

    top5 = sorted(rows, key=lambda r: (float(r["oos_profit"]), float(r["oos_ret"])), reverse=True)[:5]
    main_best = current_main_best()
    elapsed = time.time() - t0

    summary = {
        "batch": batch,
        "variants": len(rows),
        "elapsed_s": round(elapsed, 1),
        "data_hours": T,
        "data_tokens": N - (1 if D["sol_idx"] >= 0 else 0),
        "split_bar": split,
        "oos_with_trades": len(oos_tr),
        "oos_profitable": len(oos_pos),
        "oos_and_is_profitable": len(both_pos),
        "signal_counts": {k: int(v) for k, v in counts.items()},
        "family_stats": {
            k: {
                "n": v["n"],
                "with_trades": v["with_trades"],
                "pos_oos": v["pos"],
                "best_oos_profit": None if v["best_row"] is None else v["best"],
                "best_id": None if v["best_row"] is None else v["best_row"]["id"],
            }
            for k, v in fam_stats.items()
        },
        "top5": [
            {
                "id": r["id"],
                "family": r["family"],
                "entry": r["entry"],
                "exit": r["exit"],
                "sizing": r["sizing"],
                "balance": r["balance"],
                "oos_profit": r["oos_profit"],
                "oos_end": r["oos_end"],
                "oos_ret": r["oos_ret"],
                "oos_win": r["oos_win"],
                "oos_trades": r["oos_trades"],
                "oos_dd": r["oos_dd"],
                "is_ret": r["is_ret"],
                "is_trades": r["is_trades"],
            }
            for r in top5
        ],
        "main_best": None
        if main_best is None
        else {
            "id": main_best.get("id"),
            "family": main_best.get("family"),
            "oos_profit": float(main_best["oos_profit"]),
            "oos_ret": float(main_best["oos_ret"]),
            "oos_trades": int(float(main_best["oos_trades"])),
            "balance": float(main_best["balance"]),
        },
        "beat_main_best_profit": bool(
            main_best is not None and float(top5[0]["oos_profit"]) > float(main_best["oos_profit"])
        )
        if top5
        else False,
    }
    with open(ROOT / f"batch_{batch}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({k: summary[k] for k in ("batch", "variants", "elapsed_s", "oos_with_trades", "oos_profitable", "beat_main_best_profit", "top5")}, indent=2))
    print(f"wrote {out} and leaderboard.csv ({len(board)} rows)", flush=True)
    return summary


if __name__ == "__main__":
    run()
