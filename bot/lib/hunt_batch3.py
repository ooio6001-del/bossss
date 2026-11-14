#!/usr/bin/env python3
"""Batch 3 hunter: 10,000 NEW variants not logged in ideas.md.

Refines the batch 1 12h VWAP-reclaim all-in leader, diversified climax
reversal, and Bollinger-band retest; tests OR/confluence ensembles of those
signals on one shared account; and adds four fresh families (Keltner break,
z-score revert in low vol, resistance flip, green-stack momentum).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit, prange

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from solstrats import costs as K
from solstrats import engine as E
from solstrats.describe import exit_text, sizing_text
from run_backtest import prep

import hunt as H
import hunt_batch2 as H2

DATA_DIR = H.DATA_DIR
BATCH_SIZE = 10_000
NONE = H.NONE

# New exits — none of these tuples were in batch 1 or 2
EXITS = [
    (0.20, 0.25, NONE, 0.0, 96, NONE, 1.0),
    (0.60, 0.30, 0.30, 0.18, 288, NONE, 1.0),
    (1.50, 0.28, NONE, 0.0, 192, NONE, 1.0),
    (NONE, 0.40, 0.60, 0.28, 600, NONE, 1.0),
    (0.40, 0.16, 0.15, 0.12, 72, NONE, 0.45),
]
# Mix of all-in (to refine the 12h VWAP leader) and new diversified clips
# (none of these (frac, maxpos) pairs except 100%/1 were used before; 100%/1
# is reused only with new exits / new entry filters).
SIZING = [(0.03, 30), (0.07, 14), (0.12, 8), (0.18, 5), (1.00, 1)]
BALANCES = np.array([5.0, 25.0, 100.0, 250.0, 280.0], dtype=np.float64)

# Hour-of-day windows for climax overlays (not applied to VWAP — already done).
HOD_WIN = {
    0: None,
    1: (13, 16),  # US morning
    2: (0, 8),    # Asia
    3: (7, 12),   # EU
}


def _pct(x):
    return f"{x * 100:g}%"


def _hod_txt(h):
    return {1: "only during 13:00–16:00 UTC (US morning)",
            2: "only during 00:00–08:00 UTC (Asia)",
            3: "only during 07:00–12:00 UTC (EU)"}.get(int(h), "")


def entry_text(fam, p):
    if fam == "vwap_reclaim":
        s = "Buy when price reclaims its 12h VWAP after trading below it"
        if p[1]:
            s += f" on {p[1]:g}x average volume"
        if p[2]:
            s += ", close in the upper half of the hour"
        if p[3] > 0:
            s += f", reclaim of at least {_pct(p[3])}"
        if len(p) > 4 and p[4] > 0:
            s += ", only if SOL is above its 24h average"
        if len(p) > 5 and p[5] > 0:
            s += f", not more than {_pct(p[5])} above VWAP"
        if len(p) > 6 and p[6] > 0:
            s += ", confirmed still above VWAP the next hour (delayed fill)"
        return s
    if fam == "climax_reversal":
        s = (
            f"Buy the first green hour after a climax dump of {_pct(p[0])}+ "
            f"on {p[1]:g}x average volume (confirmation bar, not the dump bar)"
        )
        ht = _hod_txt(p[2])
        if ht:
            s += ", " + ht
        return s
    if fam == "bb_squeeze_break":
        return (
            f"Buy a close above the upper {int(p[0])}h Bollinger band ({p[1]:g}σ) "
            f"after a {int(p[2])}h width squeeze, only on a retest of the band "
            "(not the first poke)"
            + (f" on {p[3]:g}x average volume" if p[3] else "")
        )
    if fam == "ensemble":
        return p[-1] if isinstance(p[-1], str) else str(p)
    if fam == "keltner_break":
        s = (
            f"Buy a close above the {int(p[0])}h Keltner upper band "
            f"(EMA + {p[1]:g}×ATR)"
        )
        if p[2]:
            s += ", only after ATR compresses below 70% of its 48h average"
        return s
    if fam == "zscore_revert":
        s = (
            f"Buy a green bounce when the {int(p[0])}h close z-score is at most "
            f"−{p[1]:g}"
        )
        if p[2]:
            s += ", only when 14h ATR is below its 48h average (low-vol regime)"
        return s
    if fam == "resist_flip":
        return (
            f"Buy a bounce that holds a broken {int(p[0])}h high as support "
            f"(break within the last {int(p[1])}h, low comes within {_pct(p[2])} of the level)"
        )
    if fam == "green_stack":
        s = f"Buy the close of the {int(p[0])}th consecutive green hour"
        if p[1]:
            s += ", only if volume has risen over that streak"
        if p[2]:
            s += f", each hour up {_pct(p[2])}+"
        return s
    return fam


# Ensemble descriptions and component keys are attached as the last tuple slot.
def entries():
    """Exactly 80 entry variants, all outside the batch 1 and 2 grids."""
    Elist = []

    # --- 12h VWAP reclaim refinements (12). Window is the batch 1 leader (12h);
    # volume 0 is the leader's entry, so every vm=0 row carries a new filter.
    # Params: L=12, vm, close_upper, min_reclaim, sol_up, extend_cap, delayed.
    vwap_rows = [
        (0.0, 1, 0.0, 0, 0.0, 0),
        (0.0, 1, 0.0, 0, 0.04, 0),
        (0.0, 0, 0.0, 1, 0.0, 0),
        (0.0, 0, 0.005, 0, 0.0, 0),
        (0.0, 1, 0.01, 0, 0.0, 0),
        (1.2, 1, 0.0, 0, 0.0, 0),
        (0.0, 0, 0.0, 0, 0.0, 1),
        (0.5, 0, 0.0, 1, 0.0, 0),
        (0.0, 0, 0.0, 1, 0.04, 0),
        (1.2, 0, 0.005, 0, 0.0, 0),
        (0.0, 1, 0.0, 1, 0.0, 0),
        (0.5, 1, 0.005, 0, 0.0, 0),
    ]
    for vm, up, rec, sol, cap, delay in vwap_rows:
        Elist.append(("vwap_reclaim", (12.0, vm, up, rec, sol, cap, delay)))

    # --- diversified climax with NEW dump/vol cuts and hour-of-day overlays (12)
    # dumps/vols are not in batch 1 {8,15,25,40}%×{2,4} or batch 2 {10,12,18,20,30}%×{1.5,3,5}
    for d, vm, hod in (
        (0.09, 1.2, 0), (0.09, 2.5, 1), (0.11, 1.2, 0), (0.11, 2.5, 1),
        (0.14, 1.2, 0), (0.14, 2.5, 2), (0.16, 2.5, 0), (0.16, 1.2, 3),
        (0.22, 2.5, 0), (0.22, 6.0, 1), (0.28, 1.2, 0), (0.11, 6.0, 0),
    ):
        Elist.append(("climax_reversal", (d, vm, hod, 0.0)))

    # --- Bollinger retest only, new L/k/sq and optional volume (8)
    # not the batch 2 (20|40, 2.0, 48|96) grid
    for L, k, sq, vm in (
        (24, 1.5, 72, 0.0), (24, 2.5, 72, 0.0), (30, 2.0, 72, 0.0), (30, 2.0, 120, 0.0),
        (50, 1.5, 96, 0.0), (50, 2.5, 120, 0.0), (60, 2.0, 144, 0.0), (30, 2.5, 144, 1.5),
    ):
        Elist.append(("bb_squeeze_break", (L, k, sq, vm)))

    # --- ensembles of validated-style signals sharing one account (16)
    # Components: A=12h VWAP vm=0 (leader signal), B/C/E=sturdy climax cuts from
    # batch 2 (used only inside a combo, never standalone here), D=40h/2σ/96h retest.
    combos = [
        ("or", ("A", "B"),
         "Ensemble (shared account): 12h VWAP reclaim OR first green hour after a 10%+ dump on 1.5x volume"),
        ("or", ("A", "C"),
         "Ensemble (shared account): 12h VWAP reclaim OR first green hour after a 12%+ dump on 1.5x volume"),
        ("or", ("A", "E"),
         "Ensemble (shared account): 12h VWAP reclaim OR first green hour after an 18%+ dump on 3x volume"),
        ("or", ("A", "D"),
         "Ensemble (shared account): 12h VWAP reclaim OR 40h/2σ Bollinger retest after a 96h squeeze"),
        ("or", ("B", "D"),
         "Ensemble (shared account): 10% climax confirmation OR 40h/2σ Bollinger retest after a 96h squeeze"),
        ("or", ("C", "D"),
         "Ensemble (shared account): 12% climax confirmation OR 40h/2σ Bollinger retest after a 96h squeeze"),
        ("or", ("E", "D"),
         "Ensemble (shared account): 18% / 3x climax confirmation OR 40h/2σ Bollinger retest after a 96h squeeze"),
        ("or", ("A", "B", "D"),
         "Ensemble (shared account): 12h VWAP reclaim OR 10% climax confirmation OR 40h Bollinger retest"),
        ("or", ("A", "C", "D"),
         "Ensemble (shared account): 12h VWAP reclaim OR 12% climax confirmation OR 40h Bollinger retest"),
        ("or", ("A", "E", "D"),
         "Ensemble (shared account): 12h VWAP reclaim OR 18%/3x climax confirmation OR 40h Bollinger retest"),
        ("above_vwap", ("B",),
         "Confluence: 10% climax confirmation only while price is at or above the 12h VWAP"),
        ("above_vwap", ("C",),
         "Confluence: 12% climax confirmation only while price is at or above the 12h VWAP"),
        ("after_climax", ("A", 6),
         "Confluence: 12h VWAP reclaim only if a 10%+/1.5x climax dump occurred in the last 6h"),
        ("after_climax", ("A", 12),
         "Confluence: 12h VWAP reclaim only if a 10%+/1.5x climax dump occurred in the last 12h"),
        ("above_vwap", ("D",),
         "Confluence: 40h/2σ Bollinger retest only while price is at or above the 12h VWAP"),
        ("sol_ok", ("A", "B"),
         "Ensemble (shared account): 12h VWAP reclaim OR 10% climax confirmation, skipped if SOL is down 5%+ over 24h"),
    ]
    for i, (mode, comps, desc) in enumerate(combos):
        Elist.append(("ensemble", (mode, comps, desc)))

    # --- fresh: Keltner / ATR-channel break (8)
    for L, k, sq in (
        (20, 1.5, 0), (20, 1.5, 1), (20, 2.5, 0), (20, 2.5, 1),
        (40, 1.5, 0), (40, 1.5, 1), (40, 2.5, 0), (40, 2.5, 1),
    ):
        Elist.append(("keltner_break", (L, k, sq, 0.0)))

    # --- fresh: z-score mean reversion, optional low-vol regime (8)
    for L, z, lv in (
        (24, 1.5, 0), (24, 1.5, 1), (24, 2.5, 0), (24, 2.5, 1),
        (48, 1.5, 0), (48, 1.5, 1), (48, 2.5, 0), (48, 2.5, 1),
    ):
        Elist.append(("zscore_revert", (L, z, lv, 0.0)))

    # --- fresh: resistance-turned-support flip (8)
    for L, w, prox in (
        (24, 6, 0.01), (24, 6, 0.03), (24, 12, 0.01), (24, 12, 0.03),
        (72, 6, 0.01), (72, 6, 0.03), (72, 12, 0.01), (72, 12, 0.03),
    ):
        Elist.append(("resist_flip", (L, w, prox, 0.0)))

    # --- fresh: consecutive green-stack momentum (8)
    for n, vr, mr in (
        (3, 0, 0.0), (3, 1, 0.0), (3, 0, 0.02), (3, 1, 0.02),
        (5, 0, 0.0), (5, 1, 0.0), (6, 0, 0.0), (6, 1, 0.02),
    ):
        Elist.append(("green_stack", (n, vr, mr, 0.0)))

    if len(Elist) != 80:
        raise RuntimeError(f"expected 80 entries, got {len(Elist)}")
    return Elist


def ema(C, L):
    """Per-column EMA starting at each token's first finite close. No look-ahead."""
    T, N = C.shape
    out = np.full((T, N), np.nan)
    a = 2.0 / (L + 1.0)
    for j in range(N):
        started = False
        prev = 0.0
        for t in range(T):
            c = C[t, j]
            if not (c == c) or c <= 0:
                if started:
                    out[t, j] = prev
                continue
            if not started:
                prev = c
                out[t, j] = prev
                started = True
            else:
                prev = a * c + (1.0 - a) * prev
                out[t, j] = prev
    return out


def true_range(H, L, C):
    T, N = C.shape
    tr = np.empty((T, N))
    tr[0] = H[0] - L[0]
    prev = np.nan_to_num(C, nan=0.0)
    tr[1:] = np.maximum(H[1:] - L[1:], np.maximum(np.abs(H[1:] - prev[:-1]), np.abs(L[1:] - prev[:-1])))
    tr[:1] = np.maximum(tr[:1], 0.0)
    return tr


def rolling_z(C, L):
    mid = H.rolling_mean(C, L)
    sd = H2.rolling_std(C, L)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (C - mid) / sd
    z[~(sd > 0)] = np.nan
    return z


def mask_hod(sig, hod, win):
    if not win:
        return sig
    lo, hi = win
    return sig & ((hod >= lo) & (hod < hi))[:, None]


@njit(cache=True, parallel=True)
def sig_vwap_refine(C, V, H_, L_, first, grp, vwap, av24, sol, sol_ma,
                    vm, close_upper, min_reclaim, sol_up, extend_cap, delayed, t_hi):
    T, N = C.shape
    sig = np.zeros((T, N), dtype=np.bool_)
    for j in prange(N):
        if grp[j] == 2 or first[j] < 0:
            continue
        f0 = first[j]
        start = max(f0 + 13, 13)
        for t in range(start, t_hi):
            vw = vwap[t, j]
            prev = vwap[t - 1, j]
            c = C[t, j]
            p = C[t - 1, j]
            if not (c > 0 and p > 0 and vw == vw and prev == prev and vw > 0):
                continue
            if delayed > 0:
                # prior hour was the reclaim; this hour still holds above VWAP
                vw2 = vwap[t - 2, j] if t >= 2 else np.nan
                pp = C[t - 2, j] if t >= 2 else np.nan
                if not (vw2 == vw2 and pp == pp and pp > 0):
                    continue
                if pp >= vw2 or p < prev or c < vw:
                    continue
            else:
                if p >= prev or c < vw:
                    continue
            if c < vw * (1.0 + min_reclaim):
                continue
            if vm > 0:
                a = av24[t, j]
                if not (a == a) or a <= 0 or V[t, j] < vm * a:
                    continue
            if close_upper > 0:
                hi, lo = H_[t, j], L_[t, j]
                if not (hi == hi and lo == lo) or hi <= lo or c <= 0.5 * (hi + lo):
                    continue
            if sol_up > 0:
                s, m = sol[t], sol_ma[t]
                if not (s == s and m == m and s > 0) or s <= m:
                    continue
            if extend_cap > 0 and c > vw * (1.0 + extend_cap):
                continue
            sig[t, j] = True
    return sig


@njit(cache=True, parallel=True)
def sig_keltner(C, first, grp, mid, upper, atr, atr_ma, squeeze, t_hi):
    T, N = C.shape
    sig = np.zeros((T, N), dtype=np.bool_)
    for j in prange(N):
        if grp[j] == 2 or first[j] < 0:
            continue
        f0 = first[j]
        start = max(f0 + 2, 2)
        for t in range(start, t_hi):
            c = C[t, j]
            p = C[t - 1, j]
            up = upper[t, j]
            up1 = upper[t - 1, j]
            if not (c > 0 and p > 0 and up == up and up1 == up1):
                continue
            if c <= up or p > up1:
                continue
            if squeeze > 0:
                a, am = atr[t, j], atr_ma[t, j]
                if not (a == a and am == am and am > 0) or a >= 0.7 * am:
                    continue
            sig[t, j] = True
    return sig


@njit(cache=True, parallel=True)
def sig_zscore(C, first, grp, z, atr, atr_ma, zthr, lowvol, t_hi):
    T, N = C.shape
    sig = np.zeros((T, N), dtype=np.bool_)
    for j in prange(N):
        if grp[j] == 2 or first[j] < 0:
            continue
        f0 = first[j]
        start = max(f0 + 2, 2)
        for t in range(start, t_hi):
            zt = z[t, j]
            c = C[t, j]
            p = C[t - 1, j]
            if not (zt == zt and c > 0 and p > 0):
                continue
            if zt > -zthr:
                continue
            if c <= p:
                continue
            if lowvol > 0:
                a, am = atr[t, j], atr_ma[t, j]
                if not (a == a and am == am and am > 0) or a >= am:
                    continue
            sig[t, j] = True
    return sig


@njit(cache=True, parallel=True)
def sig_resist_flip(H_, L_, C, first, grp, Lb, W, prox, t_hi):
    T, N = C.shape
    sig = np.zeros((T, N), dtype=np.bool_)
    for j in prange(N):
        if grp[j] == 2 or first[j] < 0:
            continue
        f0 = first[j]
        start = max(f0 + Lb + W + 1, Lb + W + 1)
        for t in range(start, t_hi):
            # resistance = max high of [t-W-L, t-W)
            mx = 0.0
            ok = False
            for k in range(t - W - Lb, t - W):
                h = H_[k, j]
                if h == h and h > mx:
                    mx = h
                    ok = True
            if not ok or mx <= 0:
                continue
            # a close in (t-W, t) broke above that high
            broke = False
            for k in range(t - W, t):
                if C[k, j] > mx:
                    broke = True
                    break
            if not broke:
                continue
            c = C[t, j]
            p = C[t - 1, j]
            lo = L_[t, j]
            if not (c > 0 and p > 0 and lo == lo):
                continue
            if lo > mx * (1.0 + prox):
                continue
            if lo < mx * (1.0 - prox):
                continue
            if c <= mx or c <= p:
                continue
            sig[t, j] = True
    return sig


@njit(cache=True, parallel=True)
def sig_green_stack(C, V, first, grp, n, vol_rising, min_ret, t_hi):
    T, N = C.shape
    sig = np.zeros((T, N), dtype=np.bool_)
    for j in prange(N):
        if grp[j] == 2 or first[j] < 0:
            continue
        f0 = first[j]
        start = max(f0 + n + 1, n + 1)
        for t in range(start, t_hi):
            ok = True
            for i in range(n):
                c = C[t - i, j]
                p = C[t - i - 1, j]
                if not (c > 0 and p > 0) or c / p - 1.0 < min_ret or c <= p:
                    ok = False
                    break
            if not ok:
                continue
            if vol_rising > 0:
                for i in range(n - 1):
                    if V[t - i, j] <= V[t - i - 1, j]:
                        ok = False
                        break
                if not ok:
                    continue
            sig[t, j] = True
    return sig


@njit(cache=True)
def recent_true(src, W):
    """out[t,j] True if src[t-W+1..t] has a True (inclusive). No look-ahead."""
    T, N = src.shape
    out = np.zeros((T, N), dtype=np.bool_)
    for j in range(N):
        run = 0
        for t in range(T):
            if src[t, j]:
                run = W
            if run > 0:
                out[t, j] = True
                run -= 1
    return out


def base_signals(D, cache):
    """Validated-style component signals used by ensembles (not standalone)."""
    if "bases" in cache:
        return cache["bases"]
    C, V, H_, L_ = D["C"], D["V"], D["H"], D["L"]
    first, grp = D["first"], D["grp"]
    t_hi = D["T"] - 1
    av24 = cache["av24"]
    vwap12 = cache.setdefault(("vwap", 12), H.rolling_vwap(C, V, 12))
    A = H.sig_vwap_reclaim(C, V, first, grp, vwap12, av24, 12, 0.0, t_hi)
    B = H.sig_climax_reversal(C, V, first, grp, av24, 0.10, 1.5, t_hi)
    Csig = H.sig_climax_reversal(C, V, first, grp, av24, 0.12, 1.5, t_hi)
    Esig = H.sig_climax_reversal(C, V, first, grp, av24, 0.18, 3.0, t_hi)
    key = ("bb", 40, 2.0)
    if key not in cache:
        mid = H.rolling_mean(C, 40)
        sd = H2.rolling_std(C, 40)
        upper = mid + 2.0 * sd
        with np.errstate(invalid="ignore", divide="ignore"):
            width = (4.0 * sd) / mid
        width[~(mid > 0)] = np.nan
        cache[key] = (upper, width)
    upper, width = cache[key]
    Dsig = H2.sig_bb_squeeze(C, L_, V, first, grp, upper, width, av24, 96, 0.0, 1, t_hi)
    # climax dump bars themselves (for "after climax" confluence): previous-bar dump
    dump = H.sig_climax_reversal(C, V, first, grp, av24, 0.10, 1.5, t_hi)
    bases = {"A": A, "B": B, "C": Csig, "D": Dsig, "E": Esig, "dumpB": dump, "vwap12": vwap12}
    cache["bases"] = bases
    return bases


def make_signal(fam, p, D, cache):
    C, V, H_, L_ = D["C"], D["V"], D["H"], D["L"]
    first, grp = D["first"], D["grp"]
    t_hi = D["T"] - 1
    av24 = cache["av24"]
    if fam == "vwap_reclaim":
        Lbar = int(p[0])
        key = ("vwap", Lbar)
        if key not in cache:
            cache[key] = H.rolling_vwap(C, V, Lbar)
        sma24 = cache.setdefault(("sol_sma", 24), H.rolling_mean(D["solpx"][:, None], 24)[:, 0])
        return sig_vwap_refine(
            C, V, H_, L_, first, grp, cache[key], av24, D["solpx"], sma24,
            float(p[1]), int(p[2]), float(p[3]), int(p[4]), float(p[5]), int(p[6]), t_hi,
        )
    if fam == "climax_reversal":
        sig = H.sig_climax_reversal(C, V, first, grp, av24, float(p[0]), float(p[1]), t_hi)
        return mask_hod(sig, cache["hod"], HOD_WIN.get(int(p[2])))
    if fam == "bb_squeeze_break":
        Lbar, k, sq, vm = int(p[0]), float(p[1]), int(p[2]), float(p[3])
        key = ("bb", Lbar, k)
        if key not in cache:
            mid = H.rolling_mean(C, Lbar)
            sd = H2.rolling_std(C, Lbar)
            upper = mid + k * sd
            with np.errstate(invalid="ignore", divide="ignore"):
                width = (2.0 * k * sd) / mid
            width[~(mid > 0)] = np.nan
            cache[key] = (upper, width)
        upper, width = cache[key]
        return H2.sig_bb_squeeze(C, L_, V, first, grp, upper, width, av24, sq, vm, 1, t_hi)
    if fam == "ensemble":
        mode, comps, _desc = p[0], p[1], p[2]
        bases = base_signals(D, cache)
        if mode == "or":
            sig = np.zeros((D["T"], D["N"]), dtype=np.bool_)
            for name in comps:
                sig = sig | bases[name]
            return sig
        if mode == "above_vwap":
            src = bases[comps[0]]
            return src & (C >= bases["vwap12"]) & np.isfinite(bases["vwap12"])
        if mode == "after_climax":
            src = bases[comps[0]]
            W = int(comps[1])
            recent = cache.setdefault(("recent_dumpB", W), recent_true(bases["dumpB"], W))
            return src & recent
        if mode == "sol_ok":
            sig = np.zeros((D["T"], D["N"]), dtype=np.bool_)
            for name in comps:
                sig = sig | bases[name]
            sol = D["solpx"]
            with np.errstate(invalid="ignore", divide="ignore"):
                sr = np.full(sol.shape, np.nan)
                sr[24:] = sol[24:] / sol[:-24] - 1.0
            return sig & (sr > -0.05)[:, None]
        raise KeyError(mode)
    if fam == "keltner_break":
        Lbar, k, sq = int(p[0]), float(p[1]), int(p[2])
        mid = cache.setdefault(("ema", Lbar), ema(C, Lbar))
        atr = cache.setdefault(("atr", Lbar), H.rolling_mean(true_range(H_, L_, C), Lbar))
        atr_ma = cache.setdefault(("atrma", Lbar, 48), H.rolling_mean(true_range(H_, L_, C), 48))
        upper = mid + k * atr
        return sig_keltner(C, first, grp, mid, upper, atr, atr_ma, sq, t_hi)
    if fam == "zscore_revert":
        Lbar, zthr, lv = int(p[0]), float(p[1]), int(p[2])
        z = cache.setdefault(("z", Lbar), rolling_z(C, Lbar))
        atr = cache.setdefault(("atr", 14), H.rolling_mean(true_range(H_, L_, C), 14))
        atr_ma = cache.setdefault(("atrma", 14, 48), H.rolling_mean(true_range(H_, L_, C), 48))
        return sig_zscore(C, first, grp, z, atr, atr_ma, zthr, lv, t_hi)
    if fam == "resist_flip":
        return sig_resist_flip(H_, L_, C, first, grp, int(p[0]), int(p[1]), float(p[2]), t_hi)
    if fam == "green_stack":
        return sig_green_stack(C, V, first, grp, int(p[0]), int(p[1]), float(p[2]), t_hi)
    raise KeyError(fam)


def row_from(i, batch, fam, p, e, frac, maxpos, bal, is_, oos, full):
    r = H.row_from(i, batch, fam, p if fam != "ensemble" else (0, 0, 0, 0), e, frac, maxpos, bal, is_, oos, full)
    r["entry"] = entry_text(fam, p)
    if fam == "ensemble":
        r["params"] = json.dumps({"mode": p[0], "comps": list(p[1]), "desc": p[2]})
    else:
        r["params"] = json.dumps(list(map(float, p)))
    r["both_windows"] = H.both_windows_flag(r["oos_profit"], r["is_ret"])
    return r


def smoke_signals(D, cache):
    T, N = D["T"], D["N"]
    problems, counts = [], {}
    for fam, p in entries():
        sig = make_signal(fam, p, D, cache)
        if sig.shape != (T, N):
            problems.append(f"{fam} bad shape {sig.shape}")
            continue
        if D["sol_idx"] >= 0 and sig[:, D["sol_idx"]].any():
            problems.append(f"{fam} signaled on SOL")
        n = int(sig.sum())
        counts[fam] = counts.get(fam, 0) + n
        if n == sig.size:
            problems.append(f"{fam} all True")
        for j in range(N):
            f0 = int(D["first"][j])
            if f0 >= 0 and f0 < T and sig[f0, j]:
                problems.append(f"{fam} signal on first bar token {j}")
                break
    return problems, counts


def pack(rows, keyfn, n=5):
    out = []
    for r in sorted(rows, key=keyfn, reverse=True)[:n]:
        out.append({
            "id": r["id"], "family": r["family"], "entry": r["entry"], "exit": r["exit"],
            "sizing": r["sizing"], "balance": r["balance"],
            "oos_profit": r["oos_profit"], "oos_end": r["oos_end"], "oos_ret": r["oos_ret"],
            "oos_win": r["oos_win"], "oos_trades": r["oos_trades"], "oos_dd": r["oos_dd"],
            "is_profit": r["is_profit"], "is_ret": r["is_ret"], "is_trades": r["is_trades"],
            "is_win": r["is_win"], "is_dd": r["is_dd"],
            "full_ret": r["full_ret"], "full_trades": r["full_trades"],
            "both_windows": r["both_windows"],
            "fluke_oos_only": float(r["is_ret"]) <= 0,
        })
    return out


def run():
    ROOT.mkdir(parents=True, exist_ok=True)
    batch = H.next_batch_number()
    t0 = time.time()
    print(f"batch {batch}: loading data", flush=True)
    D = prep(DATA_DIR)
    T, N = D["T"], D["N"]
    split = D["split"]
    hod = ((D["times"].astype(np.int64) % 86400) // 3600).astype(np.int64)
    cache = {"av24": H.avg_vol(D["V"], 24), "hod": hod}
    ncfg = len(entries()) * len(EXITS) * len(SIZING) * len(BALANCES)
    print(
        f"data {T} hours x {N} tokens; split bar {split}; "
        f"{len(entries())} entries x {len(EXITS)} exits x {len(SIZING)} sizing x {len(BALANCES)} balances "
        f"= {ncfg} configs",
        flush=True,
    )
    if ncfg != BATCH_SIZE:
        raise RuntimeError(f"grid is {ncfg}, want {BATCH_SIZE}")

    print("smoke-testing signals", flush=True)
    problems, counts = smoke_signals(D, cache)
    if problems:
        raise RuntimeError("signal smoke failed: " + "; ".join(problems[:8]))
    print("signal counts by family (sum over param sets):", counts, flush=True)
    zero = [f for f, n in counts.items() if n == 0]
    if zero:
        raise RuntimeError(f"families with zero signals: {zero}")

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
        if n_sig % 10 == 0:
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
    if cfg_combo.size != BATCH_SIZE:
        raise RuntimeError(f"expanded grid is {cfg_combo.size}")

    args = (K.PCT_COST, K.FIXED_SOL, K.RENT_SOL, K.RENT_UNRECOVERED, K.IMPACT_K, K.FILL_CAP)
    res = {}
    for name, lo, hi in (("is", 0, split), ("oos", split, T), ("full", 0, T)):
        t1 = time.time()
        res[name] = E.simulate_many(
            tok, ti, to, pin, pout, D["venue"], vol_entry, D["solpx"], offs,
            cfg_combo, cfg_bal, cfg_frac, cfg_maxpos, lo, hi, *args,
        )
        print(f"sim {name} {res[name].shape} in {time.time() - t1:.1f}s", flush=True)

    rows = []
    for i in range(BATCH_SIZE):
        c = int(cfg_combo[i])
        fam, p, e = combos[c]
        s = int(cfg_size[i])
        frac, maxpos = SIZING[s]
        rows.append(row_from(i + 1, batch, fam, p, e, frac, maxpos, float(cfg_bal[i]),
                             res["is"][i], res["oos"][i], res["full"][i]))

    out = ROOT / f"batch_{batch}.csv"
    H.write_csv(out, rows)
    board = H.update_leaderboard(rows)

    oos_tr = [r for r in rows if int(r["oos_trades"]) > 0]
    oos_pos = [r for r in oos_tr if float(r["oos_profit"]) > 0]
    both_pos = [r for r in oos_pos if float(r["is_ret"]) > 0]
    both_min5 = [r for r in both_pos if int(r["oos_trades"]) >= 5 and int(r["is_trades"]) >= 5]
    flukes = [r for r in oos_pos if float(r["is_ret"]) <= 0]

    fam_stats = {}
    for r in rows:
        s = fam_stats.setdefault(r["family"], {
            "n": 0, "with_trades": 0, "pos_oos": 0, "pos_both": 0, "flukes": 0,
            "best_oos": -1e99, "best_oos_id": None, "best_both": -1e99, "best_both_id": None,
        })
        s["n"] += 1
        if int(r["oos_trades"]) > 0:
            s["with_trades"] += 1
        op, ir = float(r["oos_profit"]), float(r["is_ret"])
        if op > 0:
            s["pos_oos"] += 1
            if ir <= 0:
                s["flukes"] += 1
            else:
                s["pos_both"] += 1
                if op > s["best_both"]:
                    s["best_both"], s["best_both_id"] = op, r["id"]
        if op > s["best_oos"]:
            s["best_oos"], s["best_oos_id"] = op, r["id"]

    top5_both = pack(both_pos, lambda r: (float(r["oos_profit"]), float(r["is_profit"]), float(r["oos_ret"])))
    top5_both_min5 = pack(both_min5, lambda r: (float(r["oos_profit"]), float(r["is_profit"]), float(r["oos_ret"])))
    top5_oos = pack(rows, lambda r: (float(r["oos_profit"]), float(r["oos_ret"])))

    main_best = H.current_main_best()
    prior = [r for r in board if not r["id"].startswith(f"H{batch}-")]
    prior_both = [r for r in prior if str(r.get("both_windows", "")).lower() == "true"
                  or (float(r["oos_profit"]) > 0 and float(r["is_ret"]) > 0)]
    prior_val = max(prior_both, key=lambda r: float(r["oos_profit"])) if prior_both else None

    n_both_col = sum(1 for r in board if str(r.get("both_windows", "")).lower() == "true")
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
        "both_min5_trades": len(both_min5),
        "oos_only_flukes": len(flukes),
        "leaderboard_rows": len(board),
        "leaderboard_both_windows": n_both_col,
        "signal_counts": {k: int(v) for k, v in counts.items()},
        "family_stats": {
            k: {
                "n": v["n"], "with_trades": v["with_trades"], "pos_oos": v["pos_oos"],
                "pos_both": v["pos_both"], "flukes": v["flukes"],
                "best_oos_profit": None if v["best_oos"] < -1e98 else v["best_oos"],
                "best_oos_id": v["best_oos_id"],
                "best_both_profit": None if v["best_both"] < -1e98 else v["best_both"],
                "best_both_id": v["best_both_id"],
            }
            for k, v in fam_stats.items()
        },
        "top5_both_windows": top5_both,
        "top5_both_min5_trades": top5_both_min5,
        "top5_oos_raw": top5_oos,
        "main_best": None if main_best is None else {
            "id": main_best.get("id"), "family": main_best.get("family"),
            "oos_profit": float(main_best["oos_profit"]), "oos_ret": float(main_best["oos_ret"]),
            "oos_trades": int(float(main_best["oos_trades"])), "balance": float(main_best["balance"]),
        },
        "prior_validated_best": None if prior_val is None else {
            "id": prior_val["id"], "family": prior_val["family"],
            "oos_profit": float(prior_val["oos_profit"]), "is_ret": float(prior_val["is_ret"]),
            "oos_trades": int(prior_val["oos_trades"]), "balance": float(prior_val["balance"]),
        },
        "beat_main_best_profit_validated": bool(
            top5_both and main_best is not None
            and float(top5_both[0]["oos_profit"]) > float(main_best["oos_profit"])
        ),
        "beat_prior_validated": bool(
            top5_both and prior_val is not None
            and float(top5_both[0]["oos_profit"]) > float(prior_val["oos_profit"])
        ),
    }
    with open(ROOT / f"batch_{batch}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({
        k: summary[k] for k in (
            "batch", "variants", "elapsed_s", "oos_with_trades", "oos_profitable",
            "oos_and_is_profitable", "both_min5_trades", "oos_only_flukes",
            "leaderboard_rows", "leaderboard_both_windows",
            "beat_main_best_profit_validated", "beat_prior_validated",
            "top5_both_windows", "top5_oos_raw",
        )
    }, indent=2))
    print(f"wrote {out} and leaderboard.csv ({len(board)} rows, both_windows={n_both_col})", flush=True)
    return summary


if __name__ == "__main__":
    run()
