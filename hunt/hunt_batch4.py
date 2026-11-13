#!/usr/bin/env python3
"""Batch 4 hunter: 10,000 NEW variants + a walk-forward retest of H1-00094.

Expands the batch 3 confluence-gate idea (VWAP reclaim only after a recent
washout) to new entry/gate pairs. Adds SuperTrend, ADX-gated Donchian, FVG
reclaim, WVF washout, and Donchian-mid bounce. Does not rerun grids in ideas.md.
Walk-forward results are written separately and are not part of the 10,000.
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

ROOT = Path("/workspace/solstrats_hunt")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/workspace/solstrats")

from solstrats import costs as K
from solstrats import engine as E
from solstrats.describe import exit_text, sizing_text
from run_backtest import prep

import hunt as H
import hunt_batch2 as H2
import hunt_batch3 as H3

DATA_DIR = H.DATA_DIR
BATCH_SIZE = 10_000
NONE = H.NONE
CHI = ZoneInfo("America/Chicago")

# New exits — none of these tuples were in batches 1–3
EXITS = [
    (0.12, 0.14, NONE, 0.0, 36, NONE, 1.0),
    (0.45, 0.22, 0.25, 0.14, 144, NONE, 1.0),
    (0.90, 0.32, NONE, 0.0, 216, NONE, 1.0),
    (NONE, 0.28, 0.45, 0.22, 504, NONE, 1.0),
    (1.20, 0.18, 0.50, 0.16, 400, NONE, 0.50),
]
# New diversified sizing — none of these (frac, maxpos) pairs were used before
SIZING = [(0.02, 40), (0.09, 11), (0.11, 9), (0.14, 7), (0.16, 6)]
BALANCES = np.array([5.0, 25.0, 100.0, 250.0, 280.0], dtype=np.float64)

H1_PARAMS = (12.0, 0.0, 0.0, 0.0)
H1_EXIT = (NONE, 0.35, 0.50, 0.25, 720.0, NONE, 1.0)
H1_FRAC, H1_MAXPOS, H1_BAL = 1.00, 1, 250.0
H1_SPLIT = 2578  # original 60/40 bar


def _pct(x):
    return f"{x * 100:g}%"


def _chi(ts):
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).astimezone(CHI).strftime("%Y-%m-%d %I:%M %p %Z")


def _hod_txt(h):
    return {1: ", only during 13:00–16:00 UTC (US morning)"}.get(int(h), "")


def entry_text(fam, p):
    if fam == "confluence":
        return p[-1] if isinstance(p[-1], str) else str(p)
    if fam == "supertrend":
        s = f"Buy when SuperTrend({int(p[0])}, {p[1]:g}) flips bullish"
        if p[2]:
            s += f", only if {int(p[0])}h ADX is at least {p[2]:g}"
        return s
    if fam == "adx_donchian":
        s = (
            f"Buy a close above the prior {int(p[0])}h Donchian high "
            f"only when ADX({int(p[0])}) is at least {p[1]:g}"
        )
        if p[2]:
            s += f" on {p[2]:g}x average volume"
        return s
    if fam == "fvg_reclaim":
        s = f"Buy a reclaim of a bullish 3-bar fair-value gap formed within the last {int(p[0])}h"
        if p[1]:
            s += f" on {p[1]:g}x average volume"
        return s
    if fam == "wvf_washout":
        return (
            f"Buy the first green hour after a Williams VIX Fix({int(p[0])}) "
            f"reading of at least {_pct(p[1])} (washout wick)"
        )
    if fam == "donchian_mid":
        return (
            f"Buy a bounce back above the {int(p[0])}h Donchian mid after a pullback "
            f"within {_pct(p[1])} of the mid, while the mid has been rising for {int(p[2])}h"
        )
    return fam


def entries():
    """Exactly 80 new entry variants."""
    Elist = []

    # --- 40 confluence gates (expand H3-05689/05569; do not reuse 10%/1.5x × 6h/12h) ---
    # Each row: (entry_key, gate_key, desc)
    combos = [
        ("vwap12", "dump_12_8",
         "Confluence: 12h VWAP reclaim only if the token dumped 12%+ in an hour within the last 8h"),
        ("vwap12", "dump_12_24",
         "Confluence: 12h VWAP reclaim only if the token dumped 12%+ in an hour within the last 24h"),
        ("vwap12", "dump_20_18",
         "Confluence: 12h VWAP reclaim only if the token dumped 20%+ in an hour within the last 18h"),
        ("vwap12", "dump_20_36",
         "Confluence: 12h VWAP reclaim only if the token dumped 20%+ in an hour within the last 36h"),
        ("vwap12", "sol_4_12",
         "Confluence: 12h VWAP reclaim only if SOL is down 4%+ over the last 12h"),
        ("vwap12", "sol_8_24",
         "Confluence: 12h VWAP reclaim only if SOL is down 8%+ over the last 24h"),
        ("vwap12", "sol_8_48_hold",
         "Confluence: 12h VWAP reclaim only if SOL is down 8%+ over 48h and the token held up better"),
        ("vwap12", "climax_14_18",
         "Confluence: 12h VWAP reclaim only if a 14%+/2.5x climax dump occurred in the last 18h"),
        ("vwap12", "climax_16_24",
         "Confluence: 12h VWAP reclaim only if a 16%+/2.5x climax dump occurred in the last 24h"),
        ("vwap12", "climax_22_8",
         "Confluence: 12h VWAP reclaim only if a 22%+/2.5x climax dump occurred in the last 8h"),
        ("kel14_2", "dump_12_12",
         "Confluence: 14h Keltner (EMA+2×ATR) break only if the token dumped 12%+ in an hour within the last 12h"),
        ("kel14_2s", "dump_12_24",
         "Confluence: 14h Keltner (EMA+2×ATR) squeeze break only if the token dumped 12%+ in an hour within the last 24h"),
        ("kel30_2s", "dump_20_18",
         "Confluence: 30h Keltner (EMA+2×ATR) squeeze break only if the token dumped 20%+ in an hour within the last 18h"),
        ("kel30_3", "sol_5_12",
         "Confluence: 30h Keltner (EMA+3×ATR) break only if SOL is down 5%+ over the last 12h"),
        ("kel14_3s", "sol_8_24",
         "Confluence: 14h Keltner (EMA+3×ATR) squeeze break only if SOL is down 8%+ over the last 24h"),
        ("kel30_2", "climax_14_12",
         "Confluence: 30h Keltner (EMA+2×ATR) break only if a 14%+/2.5x climax dump occurred in the last 12h"),
        ("kel14_2", "climax_16_24",
         "Confluence: 14h Keltner (EMA+2×ATR) break only if a 16%+/2.5x climax dump occurred in the last 24h"),
        ("bb36", "sol_4_12",
         "Confluence: 36h/2σ Bollinger retest after an 80h squeeze only if SOL is down 4%+ over the last 12h"),
        ("bb36", "sol_8_24_hold",
         "Confluence: 36h/2σ Bollinger retest after an 80h squeeze only if SOL is down 8%+ over 24h and the token held up better"),
        ("bb28", "sol_5_36",
         "Confluence: 28h/1.8σ Bollinger retest after an 84h squeeze only if SOL is down 5%+ over the last 36h"),
        ("bb36", "dump_15_12",
         "Confluence: 36h/2σ Bollinger retest after an 80h squeeze only if the token dumped 15%+ in an hour within the last 12h"),
        ("bb45", "climax_14_18",
         "Confluence: 45h/2.2σ Bollinger retest after a 100h squeeze only if a 14%+/2.5x climax dump occurred in the last 18h"),
        ("climax_11", "above_kel14",
         "Confluence: 11%+/1.2x climax confirmation only while price is at or above the 14h Keltner mid (EMA)"),
        ("climax_14", "sol_4_12",
         "Confluence: 14%+/2.5x climax confirmation only if SOL is down 4%+ over the last 12h"),
        ("climax_16", "adx_below_25",
         "Confluence: 16%+/2.5x climax confirmation only when 14h ADX is below 25 (range regime)"),
        ("st10_3", "dump_12_12",
         "Confluence: SuperTrend(10, 3) bullish flip only if the token dumped 12%+ in an hour within the last 12h"),
        ("st10_3", "sol_5_24",
         "Confluence: SuperTrend(10, 3) bullish flip only if SOL is down 5%+ over the last 24h"),
        ("st14_2", "climax_14_18",
         "Confluence: SuperTrend(14, 2) bullish flip only if a 14%+/2.5x climax dump occurred in the last 18h"),
        ("z36", "sol_5_12",
         "Confluence: 36h z-score ≤ −2 bounce only if SOL is down 5%+ over the last 12h"),
        ("gs4", "dump_15_24",
         "Confluence: 4th consecutive green hour only if the token dumped 15%+ in an hour within the last 24h"),
        ("fvg24", "dump_12_12",
         "Confluence: reclaim of a 24h-aged bullish FVG only if the token dumped 12%+ in an hour within the last 12h"),
        ("vwap12", "wvf14_6",
         "Confluence: 12h VWAP reclaim only if Williams VIX Fix(14) was ≥85% within the last 6h"),
        ("kel14_2", "wvf22_12",
         "Confluence: 14h Keltner (EMA+2×ATR) break only if Williams VIX Fix(22) was ≥85% within the last 12h"),
        ("vwap12", "spring_6",
         "Confluence: 12h VWAP reclaim only if a 24h failed-breakdown spring occurred in the last 6h"),
        ("bb36", "spring_12",
         "Confluence: 36h/2σ Bollinger retest only if a 24h failed-breakdown spring occurred in the last 12h"),
        ("vwap12", "dump_12_24_us",
         "Confluence: 12h VWAP reclaim only if the token dumped 12%+ in an hour within the last 24h, only during 13:00–16:00 UTC (US morning)"),
        ("kel14_2s", "dump_12_24_us",
         "Confluence: 14h Keltner squeeze break gated by a 12%+ dump in the last 24h, only during 13:00–16:00 UTC (US morning)"),
        ("bb36", "sol_4_12_us",
         "Confluence: 36h Bollinger retest gated by SOL down 4%+ over 12h, only during 13:00–16:00 UTC (US morning)"),
        ("st10_3", "dump_20_18",
         "Confluence: SuperTrend(10, 3) bullish flip only if the token dumped 20%+ in an hour within the last 18h"),
        ("kel30_2s", "sol_8_24_hold",
         "Confluence: 30h Keltner squeeze break only if SOL is down 8%+ over 24h and the token held up better"),
    ]
    for ek, gk, desc in combos:
        hod = 1 if gk.endswith("_us") or ek.endswith("_us") else 0
        gate = gk[:-3] if gk.endswith("_us") else gk
        Elist.append(("confluence", (ek, gate, hod, desc)))

    # --- fresh SuperTrend flips (8) ---
    for atr_n, m, adx in (
        (7, 2.0, 0), (7, 3.0, 0), (7, 3.0, 20), (10, 2.0, 0),
        (10, 3.0, 0), (10, 3.0, 25), (14, 2.0, 0), (14, 3.0, 20),
    ):
        Elist.append(("supertrend", (atr_n, m, adx, 0.0)))

    # --- fresh ADX-gated Donchian (8); lookbacks are not the main-grid 24/72/168 ---
    for L, adx, vm in (
        (36, 20, 0.0), (36, 25, 0.0), (36, 25, 1.5), (48, 20, 0.0),
        (48, 25, 0.0), (48, 25, 1.5), (96, 25, 0.0), (120, 25, 1.5),
    ):
        Elist.append(("adx_donchian", (L, adx, vm, 0.0)))

    # --- fresh bullish FVG reclaim (8) ---
    for age, vm in (
        (12, 0.0), (12, 1.5), (24, 0.0), (24, 1.5),
        (48, 0.0), (48, 2.0), (72, 0.0), (24, 2.0),
    ):
        Elist.append(("fvg_reclaim", (age, vm, 0.0, 0.0)))

    # --- fresh Williams VIX Fix washout (8) ---
    for per, thr in (
        (14, 0.80), (14, 0.85), (14, 0.90), (22, 0.80),
        (22, 0.85), (22, 0.90), (14, 0.88), (22, 0.88),
    ):
        Elist.append(("wvf_washout", (per, thr, 0.0, 0.0)))

    # --- fresh Donchian mid-band bounce (8) ---
    for L, prox, rise in (
        (24, 0.01, 6), (24, 0.02, 12), (48, 0.01, 6), (48, 0.02, 12),
        (72, 0.01, 12), (72, 0.02, 24), (96, 0.02, 12), (48, 0.03, 6),
    ):
        Elist.append(("donchian_mid", (L, prox, rise, 0.0)))

    if len(Elist) != 80:
        raise RuntimeError(f"expected 80 entries, got {len(Elist)}")
    return Elist


# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------

@njit(cache=True)
def _wilder_adx(H, L, C, n):
    T = C.shape[0]
    adx = np.full(T, np.nan)
    if T < n * 2 + 2:
        return adx
    tr = np.empty(T)
    pdm = np.zeros(T)
    mdm = np.zeros(T)
    tr[0] = H[0] - L[0] if H[0] == H[0] else 0.0
    for t in range(1, T):
        up = H[t] - H[t - 1]
        dn = L[t - 1] - L[t]
        pdm[t] = up if (up > dn and up > 0) else 0.0
        mdm[t] = dn if (dn > up and dn > 0) else 0.0
        c0 = C[t - 1] if C[t - 1] == C[t - 1] else 0.0
        a = H[t] - L[t]
        b = abs(H[t] - c0)
        c = abs(L[t] - c0)
        tr[t] = a if a > b and a > c else (b if b > c else c)
        if tr[t] < 0:
            tr[t] = 0.0
    # seed
    str_ = 0.0
    sp = 0.0
    sm = 0.0
    for t in range(1, n + 1):
        str_ += tr[t]
        sp += pdm[t]
        sm += mdm[t]
    dxs = np.empty(T)
    dxs[:] = np.nan
    for t in range(n + 1, T):
        str_ = str_ - str_ / n + tr[t]
        sp = sp - sp / n + pdm[t]
        sm = sm - sm / n + mdm[t]
        if str_ <= 0:
            continue
        pdi = 100.0 * sp / str_
        mdi = 100.0 * sm / str_
        s = pdi + mdi
        dxs[t] = 0.0 if s <= 0 else 100.0 * abs(pdi - mdi) / s
    # ADX = Wilder of DX, seed with first n DX values
    acc = 0.0
    cnt = 0
    start = n + 1
    for t in range(start, start + n):
        if t < T and dxs[t] == dxs[t]:
            acc += dxs[t]
            cnt += 1
    if cnt == 0:
        return adx
    val = acc / cnt
    idx = start + n - 1
    if idx < T:
        adx[idx] = val
    for t in range(idx + 1, T):
        if dxs[t] == dxs[t]:
            val = (val * (n - 1) + dxs[t]) / n
            adx[t] = val
    return adx


def adx_grid(H_, L_, C, n):
    T, N = C.shape
    out = np.full((T, N), np.nan)
    for j in range(N):
        out[:, j] = _wilder_adx(H_[:, j], L_[:, j], C[:, j], n)
    return out


@njit(cache=True)
def _supertrend_col(H, L, C, atr, m):
    T = C.shape[0]
    trend = np.zeros(T, dtype=np.int64)
    sig = np.zeros(T, dtype=np.bool_)
    fu = 0.0
    fl = 0.0
    tr = 0
    started = False
    for t in range(T):
        if not (C[t] > 0 and atr[t] == atr[t] and atr[t] > 0 and H[t] == H[t] and L[t] == L[t]):
            if started:
                trend[t] = tr
            continue
        hl2 = 0.5 * (H[t] + L[t])
        bu = hl2 + m * atr[t]
        bl = hl2 - m * atr[t]
        if not started:
            fu, fl = bu, bl
            tr = 1 if C[t] > fu else -1
            trend[t] = tr
            started = True
            continue
        fu = bu if (bu < fu or C[t - 1] > fu) else fu
        fl = bl if (bl > fl or C[t - 1] < fl) else fl
        prev = tr
        if tr == 1:
            if C[t] < fl:
                tr = -1
        else:
            if C[t] > fu:
                tr = 1
        trend[t] = tr
        if prev == -1 and tr == 1:
            sig[t] = True
    return sig


def supertrend_sig(H_, L_, C, atr, m, first, grp, t_hi, adx=None, adx_min=0.0):
    T, N = C.shape
    sig = np.zeros((T, N), dtype=np.bool_)
    for j in range(N):
        if grp[j] == 2 or first[j] < 0:
            continue
        col = _supertrend_col(H_[:, j], L_[:, j], C[:, j], atr[:, j], m)
        sig[:t_hi, j] = col[:t_hi]
        if first[j] >= 0:
            sig[: int(first[j]) + 1, j] = False
        if adx is not None and adx_min > 0:
            for t in range(t_hi):
                if sig[t, j] and not (adx[t, j] == adx[t, j] and adx[t, j] >= adx_min):
                    sig[t, j] = False
    return sig


@njit(cache=True, parallel=True)
def sig_adx_donchian(H_, C, V, first, grp, adx, av24, Lbar, adx_min, vm, t_hi):
    T, N = C.shape
    sig = np.zeros((T, N), dtype=np.bool_)
    for j in prange(N):
        if grp[j] == 2 or first[j] < 0:
            continue
        f0 = first[j]
        start = max(f0 + Lbar + 1, Lbar + 1)
        for t in range(start, t_hi):
            a = adx[t, j]
            if not (a == a) or a < adx_min:
                continue
            mx = 0.0
            for k in range(t - Lbar, t):
                if H_[k, j] > mx:
                    mx = H_[k, j]
            c = C[t, j]
            p = C[t - 1, j]
            if not (c > 0 and mx > 0):
                continue
            if c <= mx:
                continue
            # first close through the high (avoid repeating every hour)
            if p > 0 and p > mx:
                continue
            if vm > 0:
                v = av24[t, j]
                if not (v == v) or v <= 0 or V[t, j] < vm * v:
                    continue
            sig[t, j] = True
    return sig


@njit(cache=True)
def sig_fvg_reclaim(H_, L_, C, V, first, grp, av24, max_age, vm, t_hi):
    T, N = C.shape
    sig = np.zeros((T, N), dtype=np.bool_)
    cap = 32
    for j in range(N):
        if grp[j] == 2 or first[j] < 0:
            continue
        # circular buffers of gap bottom (high of t-2) and top (low of t) and bar
        gb = np.empty(cap)
        gt = np.empty(cap)
        gs = np.empty(cap, np.int64)
        n = 0
        f0 = first[j]
        for t in range(max(f0 + 2, 2), t_hi):
            # expire old gaps
            w = 0
            for i in range(n):
                if t - gs[i] <= max_age:
                    if w != i:
                        gb[w], gt[w], gs[w] = gb[i], gt[i], gs[i]
                    w += 1
            n = w
            # new bullish FVG at t: L[t] > H[t-2]
            if L_[t, j] > H_[t - 2, j] and H_[t - 2, j] > 0:
                if n < cap:
                    gb[n] = H_[t - 2, j]
                    gt[n] = L_[t, j]
                    gs[n] = t
                    n += 1
            # reclaim: low enters a prior gap and close back above the gap bottom
            c = C[t, j]
            lo = L_[t, j]
            if c > 0 and lo == lo:
                hit = -1
                for i in range(n):
                    if gs[i] >= t:
                        continue
                    if lo <= gt[i] and c > gb[i]:
                        hit = i
                        break
                if hit >= 0:
                    ok = True
                    if vm > 0:
                        a = av24[t, j]
                        if not (a == a) or a <= 0 or V[t, j] < vm * a:
                            ok = False
                    if ok:
                        sig[t, j] = True
                    # drop the filled gap
                    for i in range(hit, n - 1):
                        gb[i], gt[i], gs[i] = gb[i + 1], gt[i + 1], gs[i + 1]
                    n -= 1
    return sig


@njit(cache=True, parallel=True)
def sig_wvf(H_, L_, C, first, grp, per, thr, t_hi):
    T, N = C.shape
    extreme = np.zeros((T, N), dtype=np.bool_)
    sig = np.zeros((T, N), dtype=np.bool_)
    for j in prange(N):
        if grp[j] == 2 or first[j] < 0:
            continue
        f0 = first[j]
        start = max(f0 + per, per)
        for t in range(start, t_hi):
            hh = 0.0
            for k in range(t - per + 1, t + 1):
                if H_[k, j] > hh:
                    hh = H_[k, j]
            if hh <= 0:
                continue
            w = (hh - L_[t, j]) / hh
            if w >= thr:
                extreme[t, j] = True
            # first green hour after an extreme on the prior bar
            if t >= start + 1 and extreme[t - 1, j]:
                c = C[t, j]
                p = C[t - 1, j]
                if c > 0 and p > 0 and c > p:
                    sig[t, j] = True
    return sig, extreme


@njit(cache=True, parallel=True)
def sig_donchian_mid(H_, L_, C, first, grp, Lbar, prox, rise, t_hi):
    T, N = C.shape
    sig = np.zeros((T, N), dtype=np.bool_)
    for j in prange(N):
        if grp[j] == 2 or first[j] < 0:
            continue
        f0 = first[j]
        start = max(f0 + Lbar + rise + 1, Lbar + rise + 1)
        for t in range(start, t_hi):
            # mid at t and t-rise
            uh = 0.0
            ll = 1e300
            for k in range(t - Lbar + 1, t + 1):
                if H_[k, j] > uh:
                    uh = H_[k, j]
                if L_[k, j] < ll:
                    ll = L_[k, j]
            if uh <= 0 or not (ll < 1e299):
                continue
            mid = 0.5 * (uh + ll)
            uh2 = 0.0
            ll2 = 1e300
            t0 = t - rise
            for k in range(t0 - Lbar + 1, t0 + 1):
                if H_[k, j] > uh2:
                    uh2 = H_[k, j]
                if L_[k, j] < ll2:
                    ll2 = L_[k, j]
            mid0 = 0.5 * (uh2 + ll2)
            if mid <= mid0:  # mid not rising
                continue
            c = C[t, j]
            p = C[t - 1, j]
            lo = L_[t, j]
            if not (c > 0 and p > 0 and lo == lo):
                continue
            # pullback into the mid zone, close back above mid
            if lo > mid * (1.0 + prox):
                continue
            if c <= mid or p > mid:
                continue
            sig[t, j] = True
    return sig


@njit(cache=True, parallel=True)
def hourly_dump(C, first, grp, dmp, t_hi):
    T, N = C.shape
    out = np.zeros((T, N), dtype=np.bool_)
    for j in prange(N):
        if grp[j] == 2 or first[j] < 0:
            continue
        f0 = first[j]
        for t in range(max(f0 + 1, 1), t_hi):
            p = C[t - 1, j]
            c = C[t, j]
            if c > 0 and p > 0 and c / p - 1.0 <= -dmp:
                out[t, j] = True
    return out


@njit(cache=True)
def sol_window_gate(sol, W, dmp):
    T = sol.shape[0]
    out = np.zeros(T, dtype=np.bool_)
    for t in range(W, T):
        s1, s0 = sol[t], sol[t - W]
        if s1 > 0 and s0 > 0 and s1 / s0 - 1.0 <= -dmp:
            out[t] = True
    return out


@njit(cache=True, parallel=True)
def token_held_up(C, sol, first, grp, W, t_hi):
    T, N = C.shape
    out = np.zeros((T, N), dtype=np.bool_)
    for j in prange(N):
        if grp[j] == 2 or first[j] < 0:
            continue
        f0 = first[j]
        for t in range(max(f0 + W, W), t_hi):
            c, b = C[t, j], C[t - W, j]
            s1, s0 = sol[t], sol[t - W]
            if c > 0 and b > 0 and s1 > 0 and s0 > 0:
                if (c / b - 1.0) > (s1 / s0 - 1.0):
                    out[t, j] = True
    return out


def wvf_extreme(H_, L_, first, grp, per, thr, t_hi):
    """Only the extreme mask from sig_wvf (for gates)."""
    # cheap: reuse sig_wvf and take extreme; C used only for confirm, pass C=H_
    # We'll compute extreme only
    T, N = H_.shape
    ext = np.zeros((T, N), dtype=np.bool_)
    for j in range(N):
        if grp[j] == 2 or first[j] < 0:
            continue
        f0 = int(first[j])
        start = max(f0 + per, per)
        for t in range(start, t_hi):
            hh = float(np.max(H_[t - per + 1 : t + 1, j]))
            if hh > 0 and (hh - L_[t, j]) / hh >= thr:
                ext[t, j] = True
    return ext


# ---------------------------------------------------------------------------
# Component / gate cache
# ---------------------------------------------------------------------------

def get_entry_signal(key, D, cache):
    C, V, H_, L_ = D["C"], D["V"], D["H"], D["L"]
    first, grp = D["first"], D["grp"]
    t_hi = D["T"] - 1
    av24 = cache["av24"]
    ck = ("entry", key)
    if ck in cache:
        return cache[ck]
    if key == "vwap12":
        vwap = cache.setdefault(("vwap", 12), H.rolling_vwap(C, V, 12))
        sig = H.sig_vwap_reclaim(C, V, first, grp, vwap, av24, 12, 0.0, t_hi)
    elif key.startswith("kel"):
        # kel14_2, kel14_2s, kel14_3s, kel30_2, kel30_2s, kel30_3
        rest = key[3:]
        squeeze = rest.endswith("s")
        rest = rest[:-1] if squeeze else rest
        Lbar, k = rest.split("_")
        Lbar, k = int(Lbar), float(k)
        mid = cache.setdefault(("ema", Lbar), H3.ema(C, Lbar))
        atr = cache.setdefault(("atr", Lbar), H.rolling_mean(H3.true_range(H_, L_, C), Lbar))
        atr_ma = cache.setdefault(("atrma48",), H.rolling_mean(H3.true_range(H_, L_, C), 48))
        upper = mid + k * atr
        sig = H3.sig_keltner(C, first, grp, mid, upper, atr, atr_ma, 1 if squeeze else 0, t_hi)
    elif key.startswith("bb"):
        # bb36, bb28, bb45
        spec = {"bb36": (36, 2.0, 80), "bb28": (28, 1.8, 84), "bb45": (45, 2.2, 100)}[key]
        Lbar, k, sq = spec
        bk = ("bb", Lbar, k)
        if bk not in cache:
            mid = H.rolling_mean(C, Lbar)
            sd = H2.rolling_std(C, Lbar)
            upper = mid + k * sd
            with np.errstate(invalid="ignore", divide="ignore"):
                width = (2.0 * k * sd) / mid
            width[~(mid > 0)] = np.nan
            cache[bk] = (upper, width)
        upper, width = cache[bk]
        sig = H2.sig_bb_squeeze(C, L_, V, first, grp, upper, width, av24, sq, 0.0, 1, t_hi)
    elif key.startswith("climax_"):
        d, vm = {"climax_11": (0.11, 1.2), "climax_14": (0.14, 2.5), "climax_16": (0.16, 2.5)}[key]
        sig = H.sig_climax_reversal(C, V, first, grp, av24, d, vm, t_hi)
    elif key.startswith("st"):
        # st10_3, st14_2
        a, m = key[2:].split("_")
        a, m = int(a), float(m)
        atr = cache.setdefault(("atr", a), H.rolling_mean(H3.true_range(H_, L_, C), a))
        sig = supertrend_sig(H_, L_, C, atr, m, first, grp, t_hi)
    elif key == "z36":
        z = cache.setdefault(("z", 36), H3.rolling_z(C, 36))
        atr = cache.setdefault(("atr", 14), H.rolling_mean(H3.true_range(H_, L_, C), 14))
        atr_ma = cache.setdefault(("atrma48",), H.rolling_mean(H3.true_range(H_, L_, C), 48))
        sig = H3.sig_zscore(C, first, grp, z, atr, atr_ma, 2.0, 0, t_hi)
    elif key == "gs4":
        sig = H3.sig_green_stack(C, V, first, grp, 4, 0, 0.0, t_hi)
    elif key == "fvg24":
        sig = sig_fvg_reclaim(H_, L_, C, V, first, grp, av24, 24, 0.0, t_hi)
    else:
        raise KeyError(key)
    cache[ck] = sig
    return sig


def get_gate(key, D, cache):
    C, V, H_, L_ = D["C"], D["V"], D["H"], D["L"]
    first, grp = D["first"], D["grp"]
    t_hi = D["T"] - 1
    av24 = cache["av24"]
    ck = ("gate", key)
    if ck in cache:
        return cache[ck]
    T, N = C.shape

    if key.startswith("dump_"):
        # dump_12_8, dump_12_24, dump_20_18, dump_15_12, ...
        parts = key.split("_")
        dmp = float(parts[1]) / 100.0
        W = int(parts[2])
        raw = cache.setdefault(("hdump", dmp), hourly_dump(C, first, grp, dmp, t_hi))
        g = H3.recent_true(raw, W)
    elif key.startswith("sol_"):
        # sol_4_12, sol_8_24, sol_8_48_hold, sol_5_12, sol_8_24_hold, sol_5_36, sol_5_24
        parts = key.split("_")
        dmp = float(parts[1]) / 100.0
        W = int(parts[2])
        hold = "hold" in parts
        sg = cache.setdefault(("solw", W, dmp), sol_window_gate(D["solpx"], W, dmp))
        g = np.broadcast_to(sg[:, None], (T, N)).copy()
        if hold:
            g &= token_held_up(C, D["solpx"], first, grp, W, t_hi)
    elif key.startswith("climax_"):
        # climax_14_18, climax_16_24, climax_22_8, climax_14_12
        parts = key.split("_")
        d = float(parts[1]) / 100.0
        W = int(parts[2])
        vm = 2.5
        src = cache.setdefault(("climax", d, vm), H.sig_climax_reversal(C, V, first, grp, av24, d, vm, t_hi))
        g = H3.recent_true(src, W)
    elif key == "above_kel14":
        mid = cache.setdefault(("ema", 14), H3.ema(C, 14))
        g = (C >= mid) & np.isfinite(mid)
    elif key == "adx_below_25":
        adx = cache.setdefault(("adx", 14), adx_grid(H_, L_, C, 14))
        g = np.isfinite(adx) & (adx < 25.0)
    elif key.startswith("wvf"):
        # wvf14_6, wvf22_12
        rest = key[3:]
        per, W = rest.split("_")
        per, W = int(per), int(W)
        ext = cache.setdefault(("wvfext", per, 0.85), wvf_extreme(H_, L_, first, grp, per, 0.85, t_hi))
        g = H3.recent_true(ext, W)
    elif key.startswith("spring_"):
        W = int(key.split("_")[1])
        src = cache.setdefault(("spring24",), H.sig_failed_breakdown(L_, H_, C, V, first, grp, av24, 24, 0.0, t_hi))
        g = H3.recent_true(src, W)
    else:
        raise KeyError(key)
    cache[ck] = g
    return g


def make_signal(fam, p, D, cache):
    C, V, H_, L_ = D["C"], D["V"], D["H"], D["L"]
    first, grp = D["first"], D["grp"]
    t_hi = D["T"] - 1
    av24 = cache["av24"]
    if fam == "confluence":
        ek, gk, hod, _ = p
        sig = get_entry_signal(ek, D, cache) & get_gate(gk, D, cache)
        if int(hod):
            h = cache["hod"]
            sig = sig & ((h >= 13) & (h <= 16))[:, None]
        return sig
    if fam == "supertrend":
        a, m, adx_min = int(p[0]), float(p[1]), float(p[2])
        atr = cache.setdefault(("atr", a), H.rolling_mean(H3.true_range(H_, L_, C), a))
        adx = cache.setdefault(("adx", a), adx_grid(H_, L_, C, a)) if adx_min > 0 else None
        return supertrend_sig(H_, L_, C, atr, m, first, grp, t_hi, adx, adx_min)
    if fam == "adx_donchian":
        Lbar, adx_min, vm = int(p[0]), float(p[1]), float(p[2])
        adx = cache.setdefault(("adx", Lbar), adx_grid(H_, L_, C, min(Lbar, 14) if Lbar >= 14 else Lbar))
        # use 14-period ADX for stability when L is long; still "ADX gated"
        if Lbar != 14:
            adx = cache.setdefault(("adx", 14), adx_grid(H_, L_, C, 14))
        return sig_adx_donchian(H_, C, V, first, grp, adx, av24, Lbar, adx_min, vm, t_hi)
    if fam == "fvg_reclaim":
        return sig_fvg_reclaim(H_, L_, C, V, first, grp, av24, int(p[0]), float(p[1]), t_hi)
    if fam == "wvf_washout":
        sig, _ = sig_wvf(H_, L_, C, first, grp, int(p[0]), float(p[1]), t_hi)
        return sig
    if fam == "donchian_mid":
        return sig_donchian_mid(H_, L_, C, first, grp, int(p[0]), float(p[1]), int(p[2]), t_hi)
    raise KeyError(fam)


# ---------------------------------------------------------------------------
# Walk-forward of H1-00094
# ---------------------------------------------------------------------------

def _sim_window(tok, ti, to, pin, pout, D, vol_entry, lo, hi, frac, maxpos, bal):
    empty = np.zeros(0)
    return E.simulate(
        tok, ti, to, pin, pout, D["venue"], vol_entry, D["solpx"], lo, hi,
        bal, frac, maxpos, K.PCT_COST, K.FIXED_SOL, K.RENT_SOL, K.RENT_UNRECOVERED,
        K.IMPACT_K, K.FILL_CAP, empty,
    )


def _fold_row(name, is_lo, is_hi, oos_hi, times, is_m, oos_m, bal):
    def pack(tag, m, lo, hi):
        return {
            f"{tag}_lo": int(lo),
            f"{tag}_hi": int(hi),
            f"{tag}_start": _chi(times[min(lo, len(times) - 1)]),
            f"{tag}_end": _chi(times[min(max(hi - 1, 0), len(times) - 1)]),
            f"{tag}_end_bal": round(float(m[0]), 4),
            f"{tag}_profit": round(float(m[0] - bal), 4),
            f"{tag}_ret": round(float(m[1]), 4),
            f"{tag}_win": round(float(m[13]), 4),
            f"{tag}_trades": int(m[4]),
            f"{tag}_dd": round(float(m[2]), 4),
            f"{tag}_sharpe": round(float(m[3]), 4),
            f"{tag}_pf": round(float(m[12]), 4),
        }
    row = {"fold": name, "balance": bal}
    row.update(pack("is", is_m, is_lo, is_hi))
    row.update(pack("oos", oos_m, is_hi, oos_hi))
    row["both_windows"] = H.both_windows_flag(row["oos_profit"], row["is_ret"])
    return row


def walk_forward(D):
    """Retest H1-00094 on many split dates. Not counted in the 10,000."""
    C, V = D["C"], D["V"]
    first, grp = D["first"], D["grp"]
    t_hi = D["T"] - 1
    T = D["T"]
    times = D["times"]
    vwap = H.rolling_vwap(C, V, 12)
    av24 = H.avg_vol(V, 24)
    sig = H.sig_vwap_reclaim(C, V, first, grp, vwap, av24, 12, 0.0, t_hi)
    tr = E.gen_trades(
        sig, np.asarray(H1_EXIT, float), D["O"], D["H"], D["L"], D["C"], D["last"], D["supply"], False
    )
    tok, ti, to, pin, pout, rs = [np.asarray(x) for x in tr]
    tok, ti, to = tok.astype(np.int64), ti.astype(np.int64), to.astype(np.int64)
    vol_entry = D["V"][np.maximum(ti - 1, 0), tok]
    n_raw = int(tok.size)

    def run_fold(name, is_lo, is_hi, oos_hi):
        is_m = _sim_window(tok, ti, to, pin, pout, D, vol_entry, is_lo, is_hi, H1_FRAC, H1_MAXPOS, H1_BAL)
        oos_m = _sim_window(tok, ti, to, pin, pout, D, vol_entry, is_hi, oos_hi, H1_FRAC, H1_MAXPOS, H1_BAL)
        return _fold_row(name, is_lo, is_hi, oos_hi, times, is_m, oos_m, H1_BAL)

    folds = []
    # original 60/40 reproduction
    folds.append(run_fold("anchored_60pct_original_split", 0, H1_SPLIT, T))
    for f in (0.40, 0.50, 0.60, 0.70, 0.80):
        mid = int(T * f)
        if mid < 24 or T - mid < 24:
            continue
        if mid == H1_SPLIT:
            continue  # already have original
        folds.append(run_fold(f"anchored_{int(f*100)}pct", 0, mid, T))

    # calendar month-end splits (America/Chicago date of the last bar of that UTC month)
    month_end_bars = {}
    for t in range(T):
        dt = datetime.fromtimestamp(float(times[t]), tz=timezone.utc)
        key = (dt.year, dt.month)
        month_end_bars[key] = t
    for (y, m), bar in sorted(month_end_bars.items()):
        mid = bar + 1  # first bar of next period
        if mid < 24 * 30 or T - mid < 24 * 14:  # need ~30d IS and ~14d OOS
            continue
        folds.append(run_fold(f"calendar_split_after_{y}-{m:02d}", 0, mid, T))

    # rolling walk-forward
    for is_len, oos_len, step, tag in (
        (1440, 720, 360, "roll_60d_30d_step15d"),
        (1080, 504, 504, "roll_45d_21d_step21d"),
        (1680, 840, 420, "roll_70d_35d_step17d"),
    ):
        k = 0
        start = 0
        while start + is_len + oos_len <= T:
            k += 1
            is_lo, is_hi, oos_hi = start, start + is_len, start + is_len + oos_len
            folds.append(run_fold(f"{tag}_{k:02d}", is_lo, is_hi, oos_hi))
            start += step

    full = _sim_window(tok, ti, to, pin, pout, D, vol_entry, 0, T, H1_FRAC, H1_MAXPOS, H1_BAL)
    orig = next(r for r in folds if r["fold"] == "anchored_60pct_original_split")

    def subset(prefix):
        rows = [r for r in folds if r["fold"].startswith(prefix) or (prefix == "anchored" and r["fold"].startswith("anchored"))]
        if prefix == "calendar":
            rows = [r for r in folds if r["fold"].startswith("calendar")]
        if prefix == "roll":
            rows = [r for r in folds if r["fold"].startswith("roll")]
        oos_pos = sum(1 for r in rows if float(r["oos_profit"]) > 0)
        both = sum(1 for r in rows if r["both_windows"] == "true")
        rets = [float(r["oos_ret"]) for r in rows]
        return {
            "n": len(rows),
            "oos_profitable": oos_pos,
            "both_windows": both,
            "oos_ret_mean": round(float(np.mean(rets)), 4) if rets else None,
            "oos_ret_median": round(float(np.median(rets)), 4) if rets else None,
            "oos_ret_min": round(float(np.min(rets)), 4) if rets else None,
            "oos_ret_max": round(float(np.max(rets)), 4) if rets else None,
            "share_oos_positive": round(oos_pos / len(rows), 4) if rows else None,
            "share_both_windows": round(both / len(rows), 4) if rows else None,
        }

    summary = {
        "strategy": "H1-00094",
        "rules": "12h VWAP reclaim, no volume filter, -35% stop, 25% trail after +50%, 720h, 100% of $250, 1 position",
        "data_hours": T,
        "data_tokens": D["N"] - (1 if D["sol_idx"] >= 0 else 0),
        "raw_trades": n_raw,
        "full_period": {
            "end": round(float(full[0]), 4),
            "ret": round(float(full[1]), 4),
            "trades": int(full[4]),
            "win": round(float(full[13]), 4),
            "dd": round(float(full[2]), 4),
        },
        "original_published": {
            "oos_profit": 538.9861, "oos_ret": 2.1559, "oos_trades": 14, "oos_win": 0.7143,
            "is_ret": 0.0906, "is_trades": 7, "split_bar": H1_SPLIT, "universe_tokens": 118,
        },
        "original_split_on_current_data": orig,
        "universe_changed": (D["N"] - (1 if D["sol_idx"] >= 0 else 0)) != 118,
        "aggregates": {
            "anchored": subset("anchored"),
            "calendar": subset("calendar"),
            "rolling": subset("roll"),
            "all_folds": subset(""),  # all: use all folds
        },
        "n_folds": len(folds),
    }
    # fix all_folds
    all_rets = [float(r["oos_ret"]) for r in folds]
    all_pos = sum(1 for r in folds if float(r["oos_profit"]) > 0)
    all_both = sum(1 for r in folds if r["both_windows"] == "true")
    summary["aggregates"]["all_folds"] = {
        "n": len(folds),
        "oos_profitable": all_pos,
        "both_windows": all_both,
        "oos_ret_mean": round(float(np.mean(all_rets)), 4),
        "oos_ret_median": round(float(np.median(all_rets)), 4),
        "oos_ret_min": round(float(np.min(all_rets)), 4),
        "oos_ret_max": round(float(np.max(all_rets)), 4),
        "share_oos_positive": round(all_pos / len(folds), 4),
        "share_both_windows": round(all_both / len(folds), 4),
    }
    # is the original 60/40 an outlier vs other anchored/rolling OOS returns?
    other = [float(r["oos_ret"]) for r in folds if r["fold"] != "anchored_60pct_original_split"]
    orig_ret = float(orig["oos_ret"])
    summary["original_vs_other_folds"] = {
        "original_oos_ret": orig_ret,
        "other_median_oos_ret": round(float(np.median(other)), 4) if other else None,
        "original_percentile_among_folds": round(
            float(np.mean([1 if orig_ret >= r else 0 for r in [float(x["oos_ret"]) for x in folds]])), 4
        ),
        "original_exceeds_other_median": bool(other and orig_ret > float(np.median(other))),
        "note": (
            "The original 60/40 OOS return is above the median of the other folds"
            if other and orig_ret > float(np.median(other))
            else "The original 60/40 OOS return is at or below the median of the other folds"
        ),
    }
    return folds, summary, n_raw


def write_wf_csv(path, folds):
    fields = list(folds[0].keys())
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(folds)


# ---------------------------------------------------------------------------
# Batch 4 grid
# ---------------------------------------------------------------------------

def row_from(i, batch, fam, p, e, frac, maxpos, bal, is_, oos, full):
    dummy = (0, 0, 0, 0) if fam == "confluence" else p
    r = H.row_from(i, batch, fam, dummy, e, frac, maxpos, bal, is_, oos, full)
    r["entry"] = entry_text(fam, p)
    if fam == "confluence":
        r["params"] = json.dumps({"entry": p[0], "gate": p[1], "hod": int(p[2]), "desc": p[3]})
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
    print(f"data {T} hours x {N} tokens; split bar {split}", flush=True)

    print("walk-forward retest of H1-00094", flush=True)
    wf_folds, wf_sum, n_raw = walk_forward(D)
    wf_csv = ROOT / "walkforward_H1-00094.csv"
    write_wf_csv(wf_csv, wf_folds)
    with open(ROOT / "walkforward_H1-00094.json", "w") as f:
        json.dump(wf_sum, f, indent=2)
    print(
        f"  folds={wf_sum['n_folds']} raw_trades={n_raw} "
        f"current_60/40 oos_ret={wf_sum['original_split_on_current_data']['oos_ret']} "
        f"published={wf_sum['original_published']['oos_ret']} "
        f"rolling both_windows={wf_sum['aggregates']['rolling']['both_windows']}/"
        f"{wf_sum['aggregates']['rolling']['n']}",
        flush=True,
    )

    ncfg = len(entries()) * len(EXITS) * len(SIZING) * len(BALANCES)
    print(
        f"{len(entries())} entries x {len(EXITS)} exits x {len(SIZING)} sizing x {len(BALANCES)} balances = {ncfg}",
        flush=True,
    )
    if ncfg != BATCH_SIZE:
        raise RuntimeError(f"grid is {ncfg}, want {BATCH_SIZE}")

    print("smoke-testing signals", flush=True)
    problems, counts = smoke_signals(D, cache)
    if problems:
        raise RuntimeError("signal smoke failed: " + "; ".join(problems[:10]))
    print("signal counts by family:", counts, flush=True)
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
    print(f"trade lists {len(combos)}  raw trades {int(lens.sum())}  median trades/list {int(np.median(lens))}", flush=True)

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
    prior_both = [r for r in prior if str(r.get("both_windows", "")).lower() == "true"]
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
        "walkforward_H1_00094": {
            "n_folds": wf_sum["n_folds"],
            "aggregates": wf_sum["aggregates"],
            "original_split_on_current_data": {
                "oos_profit": wf_sum["original_split_on_current_data"]["oos_profit"],
                "oos_ret": wf_sum["original_split_on_current_data"]["oos_ret"],
                "oos_trades": wf_sum["original_split_on_current_data"]["oos_trades"],
                "is_ret": wf_sum["original_split_on_current_data"]["is_ret"],
                "is_trades": wf_sum["original_split_on_current_data"]["is_trades"],
                "both_windows": wf_sum["original_split_on_current_data"]["both_windows"],
            },
            "original_published": wf_sum["original_published"],
            "universe_changed": wf_sum["universe_changed"],
            "original_vs_other_folds": wf_sum["original_vs_other_folds"],
            "full_period": wf_sum["full_period"],
        },
        "main_best": None if main_best is None else {
            "id": main_best.get("id"), "family": main_best.get("family"),
            "oos_profit": float(main_best["oos_profit"]), "balance": float(main_best["balance"]),
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
            "top5_both_windows", "top5_oos_raw", "walkforward_H1_00094",
        )
    }, indent=2))
    print(f"wrote {out}, leaderboard.csv ({len(board)}), {wf_csv}", flush=True)
    return summary


if __name__ == "__main__":
    run()
