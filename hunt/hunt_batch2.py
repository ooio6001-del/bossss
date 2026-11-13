#!/usr/bin/env python3
"""Batch 2 hunter: 10,000 NEW variants not logged in ideas.md.

Refines VWAP reclaim, climax reversal, and sol_risk_on in new parameter
regions with diversified sizing, and adds five fresh families from research.
Does not reuse batch 1's entry x exit x sizing grid.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

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

DATA_DIR = H.DATA_DIR
BATCH_SIZE = 10_000
NONE = H.NONE

# New exits — none of these tuples were in batch 1
EXITS = [
    (0.15, 0.12, NONE, 0.0, 48, NONE, 1.0),
    (0.35, 0.18, 0.20, 0.10, 120, NONE, 1.0),
    (0.75, 0.25, NONE, 0.0, 336, NONE, 1.0),
    (2.00, 0.40, 1.00, 0.30, 480, NONE, 1.0),
    (NONE, 0.22, 0.40, 0.20, 360, NONE, 1.0),
]
# Diversified sizing — none of these (frac, maxpos) pairs were in batch 1
SIZING = [(0.04, 25), (0.06, 15), (0.10, 10), (0.15, 6), (0.25, 4)]
BALANCES = np.array([5.0, 25.0, 100.0, 250.0, 280.0], dtype=np.float64)


def _pct(x):
    return f"{x * 100:g}%"


def entry_text(fam, p):
    if fam == "vwap_reclaim":
        s = f"Buy when price reclaims its {int(p[0])}h VWAP after trading below it"
        if p[1]:
            s += f" on {p[1]:g}x average volume"
        if p[2]:
            s += ", only during 13:00–16:00 UTC (US morning)"
        return s
    if fam == "climax_reversal":
        s = (
            f"Buy the first green hour after a climax dump of {_pct(p[0])}+ "
            f"on {p[1]:g}x average volume (confirmation bar, not the dump bar)"
        )
        if int(p[2]) == 2:
            s += ", close also above the dump bar's open"
        return s
    if fam == "sol_risk_on":
        s = f"Buy a green hour when SOL has dropped {_pct(p[0])}+ over {int(p[1])}h"
        if p[2]:
            s += " and the token has held up better than SOL"
        return s
    if fam == "stoch_uptrend":
        return (
            f"Buy when {int(p[0])}h stochastic %K crosses up through {p[1]:g} "
            f"while price is above its {int(p[2])}h average"
        )
    if fam == "bb_squeeze_break":
        s = (
            f"Buy a close above the upper {int(p[0])}h Bollinger band ({p[1]:g}σ) "
            f"after a {int(p[2])}h width squeeze"
        )
        if p[3]:
            s += ", only on a retest of the band (not the first poke)"
        return s
    if fam == "nr7_break":
        s = f"Buy a close above the high of the narrowest-range bar of the last {int(p[0])} hours"
        if p[1]:
            s += f" on {p[1]:g}x average volume"
        if p[2]:
            s += f", breakout hour up {_pct(p[2])}+"
        return s
    if fam == "inside_break":
        n = int(p[0])
        s = f"Buy a close above the mother-bar high after {n} inside bar" + ("s" if n != 1 else "")
        if p[1]:
            s += f" on {p[1]:g}x average volume"
        if p[2]:
            s += ", only if the mother bar was red"
        return s
    if fam == "us_session_orb":
        s = (
            f"Buy a close above the first-{int(p[0])}h US-morning (13:00 UTC) "
            "opening-range high"
        )
        if p[1]:
            s += f" on {p[1]:g}x average volume"
        if p[2]:
            s += ", established tokens only"
        return s
    return fam


def entries():
    """Exactly 80 entry variants, all outside the batch 1 grid."""
    P = lambda *a: tuple(list(a) + [0.0] * (4 - len(a)))
    Elist = []
    # --- refined VWAP: windows other than 12/24/48/72, new vol and US-morning filter ---
    for L, vm, hod in (
        (6, 0, 0), (6, 1.5, 0), (6, 0, 1), (6, 1.5, 1),
        (8, 1, 0), (8, 3, 0), (8, 1, 1), (8, 3, 1),
        (16, 1.5, 0), (16, 3, 0), (16, 0, 1), (16, 1.5, 1),
        (36, 1, 0), (36, 3, 0), (96, 0, 0), (168, 1.5, 0),
    ):
        Elist.append(("vwap_reclaim", P(L, vm, hod)))
    # --- refined climax: new dump/vol cuts; mode 2 = engulf the dump bar ---
    for d, vm, mode in (
        (0.10, 1.5, 1), (0.10, 3, 1), (0.10, 5, 1),
        (0.12, 1.5, 1), (0.12, 3, 1), (0.12, 5, 2),
        (0.18, 1.5, 1), (0.18, 3, 1), (0.20, 3, 1),
        (0.20, 5, 1), (0.30, 1.5, 1), (0.30, 3, 2),
    ):
        Elist.append(("climax_reversal", P(d, vm, mode)))
    # --- refined sol_risk_on: new dump depths and windows (not 3%/6% x 6h/24h) ---
    for dmp, w, hold in (
        (0.04, 3, 0), (0.04, 3, 1), (0.04, 12, 1), (0.05, 12, 0),
        (0.05, 12, 1), (0.05, 36, 1), (0.08, 12, 0), (0.08, 12, 1),
        (0.08, 48, 1), (0.10, 12, 1), (0.10, 36, 0), (0.10, 48, 1),
    ):
        Elist.append(("sol_risk_on", P(dmp, w, hold)))
    # --- fresh: stochastic oversold cross in an uptrend ---
    for per, os, ma in (
        (9, 20, 72), (9, 20, 168), (9, 30, 72), (9, 30, 168),
        (14, 20, 72), (14, 20, 168), (14, 30, 72), (14, 30, 168),
    ):
        Elist.append(("stoch_uptrend", P(per, os, ma)))
    # --- fresh: Bollinger squeeze then break, optional retest ---
    for L, k, sq, retest in (
        (20, 2.0, 48, 0), (20, 2.0, 48, 1), (20, 2.0, 96, 0), (20, 2.0, 96, 1),
        (40, 2.0, 48, 0), (40, 2.0, 48, 1), (40, 2.0, 96, 0), (40, 2.0, 96, 1),
    ):
        Elist.append(("bb_squeeze_break", P(L, k, sq, retest)))
    # --- fresh: NR5/NR7 range-contraction break ---
    for n, vm, mr in (
        (5, 0, 0.0), (5, 2, 0.0), (5, 0, 0.02), (5, 2, 0.02),
        (7, 0, 0.0), (7, 2, 0.0), (7, 0, 0.02), (7, 2, 0.02),
    ):
        Elist.append(("nr7_break", P(n, vm, mr)))
    # --- fresh: inside-bar break of the mother high ---
    for n, vm, red in (
        (1, 0, 0), (1, 0, 1), (1, 2, 0), (1, 2, 1),
        (2, 0, 0), (2, 0, 1), (2, 2, 0), (2, 2, 1),
    ):
        Elist.append(("inside_break", P(n, vm, red)))
    # --- fresh: US-morning ORB (13:00 UTC), not UTC-midnight ORB from batch 1 ---
    for k, vm, est in (
        (1, 0, 0), (1, 0, 1), (1, 1.5, 0), (1, 1.5, 1),
        (2, 0, 0), (2, 0, 1), (2, 3, 0), (2, 3, 1),
    ):
        Elist.append(("us_session_orb", P(k, vm, est)))
    if len(Elist) != 80:
        raise RuntimeError(f"expected 80 entries, got {len(Elist)}")
    return Elist


def rolling_std(C, L):
    x = np.nan_to_num(C, nan=0.0)
    mean = H.rolling_mean(C, L)
    mean2 = H._rolling_sum(x * x, L) / L
    var = np.maximum(mean2 - mean * mean, 0.0)
    out = np.sqrt(var)
    out[: L - 1] = np.nan
    return out


def stoch_k(H_, L_, C, period):
    T, N = C.shape
    k = np.full((T, N), np.nan)
    for t in range(period - 1, T):
        hh = np.max(H_[t - period + 1 : t + 1], axis=0)
        ll = np.min(L_[t - period + 1 : t + 1], axis=0)
        den = hh - ll
        with np.errstate(invalid="ignore", divide="ignore"):
            k[t] = 100.0 * (C[t] - ll) / den
        k[t, den <= 0] = np.nan
    return k


def us_opening_range_high(H_, first, times, Kbar, start_hod=13):
    T, N = H_.shape
    hod = ((times.astype(np.int64) % 86400) // 3600)
    day = times.astype(np.int64) // 86400
    orh = np.full((T, N), np.nan)
    for j in range(N):
        f0 = int(first[j])
        if f0 < 0:
            continue
        t = f0
        while t < T:
            d = day[t]
            t_end = t
            while t_end < T and day[t_end] == d:
                t_end += 1
            mx = -1e99
            cnt = 0
            formed = -1
            for u in range(t, t_end):
                if start_hod <= hod[u] < start_hod + Kbar:
                    if H_[u, j] > mx:
                        mx = H_[u, j]
                    cnt += 1
                    formed = u
            if cnt >= Kbar and mx > 0 and formed >= 0:
                for u in range(formed + 1, t_end):
                    orh[u, j] = mx
            t = t_end if t_end > t else t + 1
    return orh


@njit(cache=True, parallel=True)
def sig_climax_engulf(C, O, V, first, grp, av24, dmp, vm, t_hi):
    """Confirmation close must also be above the dump bar's open."""
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
            if c <= p:
                continue
            if c <= O[t - 1, j]:
                continue
            sig[t, j] = True
    return sig


@njit(cache=True, parallel=True)
def sig_stoch_uptrend(C, first, grp, kst, ma, os, t_hi):
    T, N = C.shape
    sig = np.zeros((T, N), dtype=np.bool_)
    for j in prange(N):
        if grp[j] == 2 or first[j] < 0:
            continue
        f0 = first[j]
        for t in range(max(f0 + 2, 2), t_hi):
            kt, kp = kst[t, j], kst[t - 1, j]
            m = ma[t, j]
            c = C[t, j]
            if not (kt == kt and kp == kp and m == m and c > 0):
                continue
            if kp >= os or kt < os:
                continue
            if c <= m:
                continue
            sig[t, j] = True
    return sig


@njit(cache=True, parallel=True)
def sig_bb_squeeze(C, L_, V, first, grp, upper, width, av24, Sq, vm, retest, t_hi):
    T, N = C.shape
    sig = np.zeros((T, N), dtype=np.bool_)
    for j in prange(N):
        if grp[j] == 2 or first[j] < 0:
            continue
        f0 = first[j]
        start = max(f0 + Sq + 2, Sq + 2)
        for t in range(start, t_hi):
            w = width[t, j]
            up = upper[t, j]
            c = C[t, j]
            if not (w == w and w > 0 and up == up and up > 0 and c > 0):
                continue
            mn = 1e300
            ok = True
            for k in range(t - Sq + 1, t + 1):
                wk = width[k, j]
                if not (wk == wk):
                    ok = False
                    break
                if wk < mn:
                    mn = wk
            if not ok or w > mn * 1.001:
                continue
            if retest == 0:
                pup = upper[t - 1, j]
                p = C[t - 1, j]
                if not (pup == pup and p == p):
                    continue
                if c <= up or p > pup:
                    continue
            else:
                saw = False
                for k in range(t - Sq, t):
                    if C[k, j] > upper[k, j]:
                        saw = True
                        break
                if not saw:
                    continue
                if L_[t, j] > up:
                    continue
                if c <= up:
                    continue
            if vm > 0:
                a = av24[t, j]
                if not (a == a) or a <= 0 or V[t, j] < vm * a:
                    continue
            sig[t, j] = True
    return sig


@njit(cache=True, parallel=True)
def sig_nr7_break(H_, L_, C, V, first, grp, av24, n, vm, mr, t_hi):
    T, N = C.shape
    sig = np.zeros((T, N), dtype=np.bool_)
    for j in prange(N):
        if grp[j] == 2 or first[j] < 0:
            continue
        f0 = first[j]
        start = max(f0 + n + 1, n + 1)
        for t in range(start, t_hi):
            # bar t-1 is the narrowest of the last n completed bars
            nr = H_[t - 1, j] - L_[t - 1, j]
            if not (nr == nr) or nr <= 0:
                continue
            is_nr = True
            for k in range(t - n, t - 1):
                rk = H_[k, j] - L_[k, j]
                if rk == rk and rk < nr:
                    is_nr = False
                    break
            if not is_nr:
                continue
            c = C[t, j]
            p = C[t - 1, j]
            if not (c > 0 and p > 0):
                continue
            if c <= H_[t - 1, j]:
                continue
            if c / p - 1.0 < mr:
                continue
            if vm > 0:
                a = av24[t, j]
                if not (a == a) or a <= 0 or V[t, j] < vm * a:
                    continue
            sig[t, j] = True
    return sig


@njit(cache=True, parallel=True)
def sig_inside_break(H_, L_, C, O, V, first, grp, av24, n_in, vm, red_mother, t_hi):
    T, N = C.shape
    sig = np.zeros((T, N), dtype=np.bool_)
    for j in prange(N):
        if grp[j] == 2 or first[j] < 0:
            continue
        f0 = first[j]
        start = max(f0 + n_in + 3, n_in + 3)
        for t in range(start, t_hi):
            # mother is bar t-1-n_in; bars t-n_in .. t-1 are successive insides
            m = t - 1 - n_in
            mh, ml = H_[m, j], L_[m, j]
            if not (mh == mh and ml == ml) or mh <= ml:
                continue
            ok = True
            prev_h, prev_l = mh, ml
            for k in range(m + 1, t):
                hk, lk = H_[k, j], L_[k, j]
                if not (hk == hk and lk == lk) or hk > prev_h or lk < prev_l:
                    ok = False
                    break
                prev_h, prev_l = hk, lk
            if not ok:
                continue
            if red_mother > 0 and C[m, j] >= O[m, j]:
                continue
            c = C[t, j]
            if not (c > 0) or c <= mh:
                continue
            if vm > 0:
                a = av24[t, j]
                if not (a == a) or a <= 0 or V[t, j] < vm * a:
                    continue
            sig[t, j] = True
    return sig


@njit(cache=True, parallel=True)
def sig_us_orb(C, V, first, grp, orh, av24, vm, est_only, t_hi):
    T, N = C.shape
    sig = np.zeros((T, N), dtype=np.bool_)
    for j in prange(N):
        if grp[j] == 2 or first[j] < 0:
            continue
        if est_only > 0 and grp[j] != 0:
            continue
        f0 = first[j]
        for t in range(max(f0 + 2, 2), t_hi):
            oh = orh[t, j]
            c = C[t, j]
            p = C[t - 1, j]
            if not (c > 0 and oh == oh and oh > 0):
                continue
            if c <= oh:
                continue
            if p > 0 and orh[t - 1, j] == orh[t - 1, j] and p > orh[t - 1, j]:
                continue
            if vm > 0:
                a = av24[t, j]
                if not (a == a) or a <= 0 or V[t, j] < vm * a:
                    continue
            sig[t, j] = True
    return sig


def make_signal(fam, p, D, cache):
    C, V, H_, L_, O = D["C"], D["V"], D["H"], D["L"], D["O"]
    first, grp = D["first"], D["grp"]
    t_hi = D["T"] - 1
    av24 = cache["av24"]
    if fam == "vwap_reclaim":
        Lbar, vm, hodf = int(p[0]), float(p[1]), int(p[2])
        key = ("vwap", Lbar)
        if key not in cache:
            cache[key] = H.rolling_vwap(C, V, Lbar)
        sig = H.sig_vwap_reclaim(C, V, first, grp, cache[key], av24, Lbar, vm, t_hi)
        if hodf:
            hod = cache["hod"]
            mask = (hod >= 13) & (hod <= 16)
            sig = sig & mask[:, None]
        return sig
    if fam == "climax_reversal":
        dmp, vm, mode = float(p[0]), float(p[1]), int(p[2])
        if mode == 2:
            return sig_climax_engulf(C, O, V, first, grp, av24, dmp, vm, t_hi)
        return H.sig_climax_reversal(C, V, first, grp, av24, dmp, vm, t_hi)
    if fam == "sol_risk_on":
        return H.sig_sol_risk_on(C, D["solpx"], first, grp, float(p[0]), int(p[1]), int(p[2]), t_hi)
    if fam == "stoch_uptrend":
        per, os, mal = int(p[0]), float(p[1]), int(p[2])
        kk = ("stoch", per)
        mk = ("ma", mal)
        if kk not in cache:
            cache[kk] = stoch_k(H_, L_, C, per)
        if mk not in cache:
            cache[mk] = H.rolling_mean(C, mal)
        return sig_stoch_uptrend(C, first, grp, cache[kk], cache[mk], os, t_hi)
    if fam == "bb_squeeze_break":
        Lbar, k, sq, retest = int(p[0]), float(p[1]), int(p[2]), int(p[3])
        key = ("bb", Lbar, k)
        if key not in cache:
            mid = H.rolling_mean(C, Lbar)
            sd = rolling_std(C, Lbar)
            upper = mid + k * sd
            with np.errstate(invalid="ignore", divide="ignore"):
                width = (2.0 * k * sd) / mid
            width[~(mid > 0)] = np.nan
            cache[key] = (upper, width)
        upper, width = cache[key]
        return sig_bb_squeeze(C, L_, V, first, grp, upper, width, av24, sq, 0.0, retest, t_hi)
    if fam == "nr7_break":
        return sig_nr7_break(H_, L_, C, V, first, grp, av24, int(p[0]), float(p[1]), float(p[2]), t_hi)
    if fam == "inside_break":
        return sig_inside_break(
            H_, L_, C, O, V, first, grp, av24, int(p[0]), float(p[1]), int(p[2]), t_hi
        )
    if fam == "us_session_orb":
        Kbar, vm, est = int(p[0]), float(p[1]), int(p[2])
        key = ("usorb", Kbar)
        if key not in cache:
            cache[key] = us_opening_range_high(H_, first, D["times"], Kbar, 13)
        return sig_us_orb(C, V, first, grp, cache[key], av24, vm, est, t_hi)
    raise KeyError(fam)


def row_from(i, batch, fam, p, e, frac, maxpos, bal, is_, oos, full):
    r = H.row_from(i, batch, fam, p, e, frac, maxpos, bal, is_, oos, full)
    r["entry"] = entry_text(fam, p)  # override hunt.py text for new / refined params
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
        rows.append(row_from(i + 1, batch, fam, p, e, frac, maxpos, float(cfg_bal[i]), res["is"][i], res["oos"][i], res["full"][i]))

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
    # best validated from prior leaderboard (IS>0, OOS>0)
    prior_val = None
    if (ROOT / "leaderboard.csv").exists() or True:
        prior = [r for r in board if int(r.get("batch", 0) or r["id"][1:2] or 0) != batch]
        # board now includes this batch; use only earlier ids
        prior = [r for r in board if not r["id"].startswith(f"H{batch}-")]
        prior_both = [r for r in prior if float(r["oos_profit"]) > 0 and float(r["is_ret"]) > 0]
        if prior_both:
            prior_val = max(prior_both, key=lambda r: float(r["oos_profit"]))

    beat_main = bool(both_pos and main_best is not None and float(both_pos and top5_both[0]["oos_profit"]) > float(main_best["oos_profit"])) if top5_both and main_best else False
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
        "beat_main_best_profit_validated": beat_main,
        "beat_prior_validated": bool(
            top5_both and prior_val is not None and float(top5_both[0]["oos_profit"]) > float(prior_val["oos_profit"])
        ),
    }
    with open(ROOT / f"batch_{batch}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({
        k: summary[k] for k in (
            "batch", "variants", "elapsed_s", "oos_with_trades", "oos_profitable",
            "oos_and_is_profitable", "both_min5_trades", "oos_only_flukes",
            "beat_main_best_profit_validated", "beat_prior_validated",
            "top5_both_windows", "top5_oos_raw",
        )
    }, indent=2))
    print(f"wrote {out} and leaderboard.csv ({len(board)} rows)", flush=True)
    return summary


if __name__ == "__main__":
    run()
