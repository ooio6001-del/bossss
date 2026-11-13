#!/usr/bin/env python3
"""Batch 6 hunter: 10,000 NEW variants + walk-forward of the new top 20
and of the main-sweep moonshot leaders.

Gates unused entries (Double-N, green-stack, Bollinger retest) by a token
dump, and adds Williams %R, CCI hook, Hull MA flip, and pin-bar families.
Walk-forward is not part of the 10,000. Updates robust_leaderboard.csv.
"""
from __future__ import annotations

import json
import re
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

import hunt as H
import hunt_batch2 as H2
import hunt_batch3 as H3
import hunt_batch4 as H4
import hunt_batch5 as H5
import wf_lib as WF

DATA_DIR = H.DATA_DIR
BATCH_SIZE = 10_000
NONE = H.NONE

EXITS = [
    (0.25, 0.16, NONE, 0.0, 48, NONE, 1.0),
    (0.70, 0.21, 0.30, 0.14, 120, NONE, 1.0),
    (1.40, 0.26, NONE, 0.0, 288, NONE, 1.0),
    (NONE, 0.30, 0.60, 0.20, 384, NONE, 1.0),
    (2.00, 0.18, 0.42, 0.17, 216, NONE, 1.0),
]
SIZING = [(0.045, 22), (0.065, 16), (0.095, 10), (0.155, 6), (0.28, 3)]
BALANCES = np.array([35.0, 55.0, 85.0, 175.0, 320.0], dtype=np.float64)


def entry_text(fam, p):
    if fam == "washout_gate":
        return p[-1] if isinstance(p[-1], str) else str(p)
    if fam == "williams_r":
        return (
            f"Buy when Williams %R({int(p[0])}) crosses up from below {p[1]:g}"
        )
    if fam == "cci_hook":
        return f"Buy when CCI({int(p[0])}) crosses up from below {p[1]:g}"
    if fam == "hull_flip":
        n, slow = int(p[0]), int(p[1])
        if slow <= 0:
            return f"Buy when Hull MA({n}) turns up"
        return f"Buy when Hull MA({n}) crosses up through Hull MA({slow})"
    if fam == "pin_bar":
        s = (
            f"Buy a bullish pin bar (lower wick at least {p[0]*100:g}% of the range, "
            f"close in the top {p[1]*100:g}%)"
        )
        if p[2] > 0:
            s += f", only if the range is at least {p[2]:g}×ATR(14)"
        return s
    return f"{fam} {p}"


def _dn_label(n, sma):
    return f"first hour of a {n}-bar closing low above its {sma}h SMA"


def _gs_label(n, vr, mr):
    s = f"close of {n} consecutive green hours"
    if mr > 0:
        s += f", each up {mr*100:g}%+"
    if vr:
        s += ", with rising volume"
    return s


def _bb_label(L, k, sq):
    return f"{L}h/{k:g}σ Bollinger retest after a {sq}h squeeze"


def entries():
    Elist = []
    dn = [
        (8, 96, "dump_9_12"), (8, 192, "dump_11_18"), (10, 120, "dump_13_24"),
        (10, 288, "dump_16_36"), (14, 96, "dump_22_16"), (14, 200, "dump_9_18"),
        (18, 120, "dump_11_24"), (20, 288, "dump_13_36"), (8, 120, "dump_16_16"),
        (10, 96, "dump_22_24"), (14, 288, "dump_9_36"), (18, 192, "dump_11_16"),
        (20, 96, "dump_13_18"), (8, 288, "dump_16_24"), (18, 96, "dump_22_36"),
        (20, 200, "dump_9_16"),
    ]
    for n, sma, gk in dn:
        ek = f"dn{n}_{sma}"
        desc = f"Washout-gate: {_dn_label(n, sma)} only if {_gate_label(gk)}"
        Elist.append(("washout_gate", (ek, gk, desc)))

    gs = [
        (4, 0, 0.01, "dump_9_12"), (4, 1, 0.01, "dump_11_18"),
        (4, 0, 0.03, "dump_13_24"), (4, 1, 0.03, "dump_16_36"),
        (7, 0, 0.01, "dump_22_16"), (5, 0, 0.015, "dump_9_18"),
        (7, 0, 0.04, "dump_11_24"), (5, 0, 0.025, "dump_13_36"),
        (8, 0, 0.01, "dump_16_16"), (5, 0, 0.012, "dump_9_24"),
        (8, 0, 0.03, "dump_9_36"), (6, 0, 0.015, "dump_11_16"),
        (4, 0, 0.04, "dump_13_18"), (7, 0, 0.03, "dump_16_24"),
        (5, 0, 0.02, "dump_9_48"), (4, 1, 0.04, "dump_9_16"),
    ]
    for n, vr, mr, gk in gs:
        ek = f"gs{n}_{vr}_{mr}"
        desc = f"Washout-gate: {_gs_label(n, vr, mr)} only if {_gate_label(gk)}"
        Elist.append(("washout_gate", (ek, gk, desc)))

    bb = [
        (16, 2.0, 60, "dump_9_12"), (18, 2.2, 80, "dump_9_48"),
        (32, 1.8, 64, "dump_13_24"), (36, 2.0, 72, "dump_9_48"),
        (42, 2.0, 90, "dump_9_48"), (48, 1.6, 100, "dump_9_18"),
        (22, 2.3, 56, "dump_11_24"), (28, 1.7, 84, "dump_13_36"),
        (16, 2.0, 48, "dump_9_36"), (18, 1.8, 60, "dump_22_24"),
        (32, 2.2, 108, "dump_9_36"), (36, 2.0, 64, "dump_11_16"),
        (42, 1.6, 100, "dump_13_18"), (48, 1.8, 72, "dump_9_48"),
        (22, 1.8, 84, "dump_22_36"), (28, 2.2, 56, "dump_9_16"),
    ]
    for L, k, sq, gk in bb:
        ek = f"bb{L}_{k}_{sq}"
        desc = f"Washout-gate: {_bb_label(L, k, sq)} only if {_gate_label(gk)}"
        Elist.append(("washout_gate", (ek, gk, desc)))

    for p in ((14, -80), (14, -90), (21, -80), (21, -90), (10, -80), (10, -90), (28, -85), (7, -80)):
        Elist.append(("williams_r", p))
    for p in ((20, -100), (20, -150), (14, -100), (14, -200), (30, -100), (30, -150), (10, -100), (20, -80)):
        Elist.append(("cci_hook", p))
    for p in ((9, 0), (16, 0), (25, 0), (49, 0), (9, 25), (16, 49), (12, 36), (20, 64)):
        Elist.append(("hull_flip", p))
    for p in ((0.60, 0.33, 0.0), (0.66, 0.30, 0.0), (0.50, 0.25, 0.0), (0.60, 0.33, 1.0),
              (0.70, 0.40, 0.0), (0.55, 0.30, 0.8), (0.65, 0.25, 1.2), (0.60, 0.35, 0.0)):
        Elist.append(("pin_bar", p))

    if len(Elist) != 80:
        raise RuntimeError(f"expected 80 entries, got {len(Elist)}")
    return Elist


def _gate_label(key):
    _, pct, win = key.split("_")
    return f"the token dumped {pct}%+ in an hour within the last {win}h"


def rolling_max(X, n):
    T, N = X.shape
    out = np.full((T, N), np.nan, dtype=np.float64)
    for t in range(n - 1, T):
        out[t] = np.max(X[t - n + 1 : t + 1], axis=0)
    return out


def wma(X, n):
    X = np.asarray(X, dtype=np.float64)
    T, N = X.shape
    out = np.full((T, N), np.nan, dtype=np.float64)
    w = np.arange(1, n + 1, dtype=np.float64)
    ws = float(w.sum())
    for t in range(n - 1, T):
        out[t] = (X[t - n + 1 : t + 1] * w[:, None]).sum(axis=0) / ws
    return out


def hma(X, n):
    n = int(n)
    n2 = max(n // 2, 1)
    ns = max(int(round(n ** 0.5)), 1)
    return wma(2.0 * wma(X, n2) - wma(X, n), ns)


def cci_grid(H_, L_, C, n):
    tp = (H_ + L_ + C) / 3.0
    mid = H.rolling_mean(tp, n)
    dev = np.abs(tp - mid)
    mad = H.rolling_mean(dev, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (tp - mid) / (0.015 * mad)
        out[~(mad > 0)] = np.nan
    return out


def williams_grid(H_, L_, C, n):
    hh = rolling_max(H_, n)
    ll = H5.rolling_min(L_, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = -100.0 * (hh - C) / (hh - ll)
        out[~(hh > ll)] = np.nan
    return out


@njit(cache=True, parallel=True)
def sig_cross_up(ind, first, grp, thresh, t_hi):
    T, N = ind.shape
    out = np.zeros((T, N), dtype=np.bool_)
    for j in prange(N):
        if grp[j] == 2 or first[j] < 0:
            continue
        f0 = int(first[j])
        start = max(f0 + 2, 2)
        for t in range(start, t_hi):
            v, v1 = ind[t, j], ind[t - 1, j]
            if v == v and v1 == v1 and v1 <= thresh and v > thresh:
                out[t, j] = True
    return out


@njit(cache=True, parallel=True)
def sig_hull_flip(fast, slow, first, grp, mode, t_hi):
    T, N = fast.shape
    out = np.zeros((T, N), dtype=np.bool_)
    for j in prange(N):
        if grp[j] == 2 or first[j] < 0:
            continue
        f0 = int(first[j])
        start = max(f0 + 2, 2)
        for t in range(start, t_hi):
            f, f1 = fast[t, j], fast[t - 1, j]
            if not (f == f and f1 == f1):
                continue
            if mode == 0:
                if f > f1 and not (fast[t - 1, j] > fast[t - 2, j] if t >= 2 and fast[t - 2, j] == fast[t - 2, j] else False):
                    # first bar of an upturn
                    prev_up = False
                    if t >= 2:
                        f2 = fast[t - 2, j]
                        prev_up = f2 == f2 and f1 > f2
                    if f > f1 and not prev_up:
                        out[t, j] = True
            else:
                s, s1 = slow[t, j], slow[t - 1, j]
                if s == s and s1 == s1 and f > s and f1 <= s1:
                    out[t, j] = True
    return out


@njit(cache=True, parallel=True)
def sig_pin_bar(O, H_, L_, C, atr, first, grp, wick, close_frac, atr_mult, t_hi):
    T, N = C.shape
    out = np.zeros((T, N), dtype=np.bool_)
    for j in prange(N):
        if grp[j] == 2 or first[j] < 0:
            continue
        f0 = int(first[j])
        start = max(f0 + 2, 2)
        for t in range(start, t_hi):
            o, h, l, c = O[t, j], H_[t, j], L_[t, j], C[t, j]
            if not (o == o and h == h and l == l and c == c and h > l and c > 0):
                continue
            rng = h - l
            if rng <= 0:
                continue
            lower = min(o, c) - l
            if lower / rng < wick:
                continue
            if (c - l) / rng < (1.0 - close_frac):
                continue
            if atr_mult > 0:
                a = atr[t, j]
                if not (a == a and a > 0 and rng >= atr_mult * a):
                    continue
            out[t, j] = True
    return out


def get_entry_signal(key, D, cache):
    C, V, H_, L_ = D["C"], D["V"], D["H"], D["L"]
    first, grp = D["first"], D["grp"]
    t_hi = D["T"] - 1
    ck = ("entry6", key)
    if ck in cache:
        return cache[ck]
    if key.startswith("dn"):
        n, sma = key[2:].split("_")
        n, sma = int(n), int(sma)
        rmin = cache.setdefault(("rmin", n), H5.rolling_min(C, n))
        sma_a = cache.setdefault(("sma", sma), H.rolling_mean(C, sma))
        sig = H5.sig_double_n(C, rmin, sma_a, first, grp, t_hi)
    elif key.startswith("gs"):
        # gs4_1_0.01
        rest = key[2:]
        n_s, vr_s, mr_s = rest.split("_", 2)
        n, vr, mr = int(n_s), int(float(vr_s)), float(mr_s)
        sig = H3.sig_green_stack(C, V, first, grp, n, vr, mr, t_hi)
    elif key.startswith("bb"):
        rest = key[2:]
        L_s, k_s, sq_s = rest.split("_")
        Lbar, k, sq = int(L_s), float(k_s), int(sq_s)
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
        av24 = cache["av24"]
        sig = H2.sig_bb_squeeze(C, L_, V, first, grp, upper, width, av24, sq, 0.0, 1, t_hi)
    else:
        raise KeyError(key)
    cache[ck] = sig
    return sig


def get_gate(key, D, cache):
    return H5.get_gate(key, D, cache)


def make_signal(fam, p, D, cache):
    C, H_, L_ = D["C"], D["H"], D["L"]
    first, grp = D["first"], D["grp"]
    t_hi = D["T"] - 1
    if fam == "washout_gate":
        return get_entry_signal(p[0], D, cache) & get_gate(p[1], D, cache)
    if fam == "williams_r":
        n, thr = int(p[0]), float(p[1])
        ind = cache.setdefault(("willr", n), williams_grid(H_, L_, C, n))
        return sig_cross_up(ind, first, grp, thr, t_hi)
    if fam == "cci_hook":
        n, thr = int(p[0]), float(p[1])
        ind = cache.setdefault(("cci", n), cci_grid(H_, L_, C, n))
        return sig_cross_up(ind, first, grp, thr, t_hi)
    if fam == "hull_flip":
        n, slow = int(p[0]), int(p[1])
        fast = cache.setdefault(("hma", n), hma(C, n))
        if slow <= 0:
            dummy = fast
            return sig_hull_flip(fast, dummy, first, grp, 0, t_hi)
        sm = cache.setdefault(("hma", slow), hma(C, slow))
        return sig_hull_flip(fast, sm, first, grp, 1, t_hi)
    if fam == "pin_bar":
        wick, cf, am = float(p[0]), float(p[1]), float(p[2])
        atr = cache.setdefault(("atr", 14), H.rolling_mean(H3.true_range(H_, L_, C), 14))
        return sig_pin_bar(D["O"], H_, L_, C, atr, first, grp, wick, cf, am, t_hi)
    raise KeyError(fam)


def row_from(i, batch, fam, p, e, frac, maxpos, bal, oos, is_, full, split, T):
    oos_profit = float(oos[0] - bal)
    is_ret = float(is_[1])
    if fam == "washout_gate":
        params = json.dumps({"entry": p[0], "gate": p[1], "desc": p[2]})
    else:
        params = json.dumps(list(map(float, p)))
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
        "params": params,
        "exit_params": json.dumps(list(map(float, e))),
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


def pack(rows, n=12, min_trades=8):
    solid = [r for r in rows if int(r["oos_trades"]) >= min_trades]
    return [
        {
            "id": r["id"],
            "family": r["family"],
            "entry": r["entry"],
            "exit": r["exit"],
            "sizing": r["sizing"],
            "balance": r["balance"],
            "oos_profit": r["oos_profit"],
            "oos_ret": r["oos_ret"],
            "oos_win": r["oos_win"],
            "oos_trades": r["oos_trades"],
            "oos_dd": r["oos_dd"],
            "oos_sharpe": r["oos_sharpe"],
            "is_ret": r["is_ret"],
            "is_trades": r["is_trades"],
            "full_ret": r["full_ret"],
            "both_windows": r["both_windows"],
        }
        for r in solid[:n]
    ]


def slim_sum(r):
    return {
        "id": r["id"],
        "batch": r["batch"],
        "family": r["family"],
        "entry": r["entry"],
        "exit": r["exit"],
        "sizing": r["sizing"],
        "balance": r["balance"],
        "passed": r["passed"],
        "source": r.get("_source", ""),
        "wf_oos_profitable_pct": r["wf_oos_profitable_pct"],
        "wf_median_oos_ret": r["wf_median_oos_ret"],
        "wf_roll_oos_profitable_pct": r["wf_roll_oos_profitable_pct"],
        "wf_roll_median_oos_ret": r["wf_roll_median_oos_ret"],
        "wf_full_ret": r["wf_full_ret"],
        "wf_full_trades": r["wf_full_trades"],
        "orig_oos_profit": r["orig_oos_profit"],
        "orig_is_ret": r["orig_is_ret"],
        "both_windows": r["both_windows"],
    }


def load_moonshot_rows():
    path = ROOT / "moonshot_wf_configs.json"
    rows = json.loads(path.read_text())
    out = []
    for r in rows:
        rec = dict(r)
        rec["batch"] = "sweep"
        rec["family"] = "moonshot"
        rec["one_shot"] = True
        rec["_source"] = r.get("_source", "sweep")
        out.append(rec)
    return out


def merge_robust(old_rows, new_summaries):
    merged = {r["id"]: r for r in old_rows}
    for r in new_summaries:
        slim = {k: r[k] for k in WF.ROBUST_FIELDS if k in r}
        slim["passed"] = r["passed"]
        if str(r.get("passed", "")).lower() == "true":
            merged[r["id"]] = slim
        elif r["id"] in merged:
            del merged[r["id"]]
    passed = [r for r in merged.values() if str(r.get("passed", "")).lower() == "true"]
    passed.sort(key=WF.robustness_key, reverse=True)
    return passed


def run_walkforwards(D, batch):
    split = int(D.get("split", int(D["T"] * 0.60)))
    b6_path = ROOT / f"batch_{batch}.csv"
    b6 = H.load_csv(b6_path)
    top20 = WF.top_both_window(b6, 20)
    for r in top20:
        r["_source"] = "batch6_top20_both"
    moons = load_moonshot_rows()
    print(f"walk-forward: {len(top20)} batch-6 both-window + {len(moons)} moonshot configs", flush=True)

    # Replay moonshot leaders at the original 60/40 split on current data
    caches = {}
    for row in moons:
        if "1000pct" in row.get("_source", "") or "10000pct" in row.get("_source", "") or row is moons[0]:
            sig = WF.signal_from_row(row, D, caches)
            e = np.asarray(json.loads(row["exit_params"]), float)
            tr = E.gen_trades(sig, e, D["O"], D["H"], D["L"], D["C"], D["last"], D["supply"], True)
            tok, ti, to, pin, pout, rs = [np.asarray(x) for x in tr]
            frac, maxpos = WF.parse_sizing(row["sizing"])
            bal = float(row["balance"])
            vol = D["V"][np.maximum(ti.astype(np.int64) - 1, 0), tok.astype(np.int64)]
            oos = WF._sim_window(
                tok.astype(np.int64), ti.astype(np.int64), to.astype(np.int64),
                pin, pout, D, vol, split, D["T"], frac, maxpos, bal,
            )
            print(
                f"  replay {row['id']} {row.get('_source')}: published oos_profit={row['oos_profit']} "
                f"replay={round(float(oos[0]-bal),4)} trades_pub={row['oos_trades']} "
                f"replay_tr={int(oos[4])}",
                flush=True,
            )

    shortlist = top20 + moons
    folds, summaries = WF.walk_forward_shortlist(D, shortlist, original_split=split)
    ranked = sorted(summaries, key=WF.robustness_key, reverse=True)
    WF.write_fold_csv(ROOT / "walkforward_batch6_folds.csv", folds)
    slim = [{k: v for k, v in r.items() if k != "_summary"} for r in ranked]
    WF.write_summary_csv(ROOT / "walkforward_batch6.csv", slim)

    old = H.load_csv(ROOT / "robust_leaderboard.csv") if (ROOT / "robust_leaderboard.csv").exists() else []
    passed = merge_robust(old, slim)
    WF.write_summary_csv(ROOT / "robust_leaderboard.csv", passed, fields=WF.ROBUST_FIELDS)

    def roll_pass(r):
        return (
            float(r["wf_full_ret"]) > 0
            and float(r["wf_roll_median_oos_ret"]) > 0
            and float(r["wf_roll_oos_profitable_pct"]) >= 50
            and int(float(r.get("wf_folds_with_oos_trades") or 0)) >= 10
        )

    b6s = [r for r in ranked if str(r["id"]).startswith("H6-") or str(r["batch"]) == "6"]
    ms = [r for r in ranked if r["family"] == "moonshot" or str(r["id"]).startswith("MS-")]
    ms_official = [r for r in ms if r["passed"] == "true"]
    ms_roll = [r for r in ms if roll_pass(r)]
    payload = {
        "pass_rule": WF.pass_rule_text(),
        "n_batch6_top20": len(b6s),
        "n_moonshot": len(ms),
        "n_moonshot_passed_official": len(ms_official),
        "n_moonshot_passed_rolling": len(ms_roll),
        "any_moonshot_official_pass": len(ms_official) > 0,
        "any_moonshot_rolling_pass": len(ms_roll) > 0,
        "batch6_top20_by_rolling": [slim_sum(r) for r in b6s],
        "moonshot_by_rolling": [slim_sum(r) for r in ms],
        "moonshot_official_passers": [slim_sum(r) for r in ms_official],
        "moonshot_rolling_passers": [slim_sum(r) for r in ms_roll],
        "n_robust_board": len(passed),
        "new_passers": [slim_sum(r) for r in ranked if r["passed"] == "true"],
    }
    with open(ROOT / "walkforward_batch6.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(
        f"walk-forward done: b6={len(b6s)} moon={len(ms)} "
        f"moon_official_pass={len(ms_official)} moon_rolling_pass={len(ms_roll)} "
        f"robust_board={len(passed)}",
        flush=True,
    )
    return payload


def run():
    ROOT.mkdir(parents=True, exist_ok=True)
    batch = H.next_batch_number()
    t0 = time.time()
    print(f"batch {batch}: loading data", flush=True)
    D = H.prep(DATA_DIR)
    T, N = D["T"], D["N"]
    split = D["split"]
    cache = {
        "av24": H.avg_vol(D["V"], 24),
        "hod": ((D["times"].astype(np.int64) % 86400) // 3600).astype(np.int64),
    }
    print(f"data {T} hours x {N} tokens; split bar {split}", flush=True)

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
        raise RuntimeError("signal smoke failed: " + "; ".join(problems[:12]))
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
        print(f"  {name} simulated in {time.time() - t1:.1f}s", flush=True)

    rows = []
    for i in range(BATCH_SIZE):
        c = int(cfg_combo[i])
        fam, p, e = combos[c]
        rows.append(
            row_from(
                i, batch, fam, p, e, float(cfg_frac[i]), int(cfg_maxpos[i]),
                float(cfg_bal[i]), res["oos"][i], res["is"][i], res["full"][i], split, T,
            )
        )

    out_csv = ROOT / f"batch_{batch}.csv"
    H.write_csv(out_csv, rows)
    all_rows = H.update_leaderboard(rows)

    oos_pos = [r for r in rows if float(r["oos_profit"]) > 0]
    both = [r for r in rows if r["both_windows"] == "true"]
    flukes = [r for r in oos_pos if r["both_windows"] != "true"]
    both_solid = [r for r in both if int(r["oos_trades"]) >= 8 and int(r["is_trades"]) >= 4]
    oos_pos.sort(key=lambda r: (float(r["oos_profit"]), float(r["oos_ret"])), reverse=True)
    both.sort(key=lambda r: (float(r["oos_profit"]), float(r["is_profit"])), reverse=True)
    both_solid.sort(key=lambda r: (float(r["oos_profit"]), float(r["is_profit"])), reverse=True)
    flukes.sort(key=lambda r: float(r["oos_profit"]), reverse=True)

    by_fam = {}
    for r in rows:
        by_fam.setdefault(r["family"], {"n": 0, "oos_pos": 0, "both": 0})
        by_fam[r["family"]]["n"] += 1
        if float(r["oos_profit"]) > 0:
            by_fam[r["family"]]["oos_pos"] += 1
        if r["both_windows"] == "true":
            by_fam[r["family"]]["both"] += 1

    prior = [r for r in all_rows if str(r["batch"]).isdigit() and int(r["batch"]) < batch]
    prior_both = [r for r in prior if r.get("both_windows") == "true"]
    prior_both.sort(key=lambda r: (float(r["oos_profit"]), float(r.get("is_profit") or 0)), reverse=True)
    prior_best = prior_both[0] if prior_both else None

    summary = {
        "batch": batch,
        "n": len(rows),
        "seconds": round(time.time() - t0, 1),
        "data": {"hours": int(T), "tokens": int(N), "split_bar": int(split)},
        "oos_profitable": len(oos_pos),
        "both_windows_profitable": len(both),
        "both_windows_solid": len(both_solid),
        "oos_only_flukes": len(flukes),
        "by_family": by_fam,
        "top_both_windows": pack(both, 12, 0),
        "top_both_windows_solid": pack(both_solid, 12, 8),
        "top_oos": pack(oos_pos, 12, 0),
        "top_flukes": pack(flukes, 8, 0),
        "best_prior_both_windows": (
            {
                "id": prior_best["id"],
                "family": prior_best["family"],
                "entry": prior_best["entry"],
                "oos_profit": prior_best["oos_profit"],
                "oos_ret": prior_best["oos_ret"],
                "is_ret": prior_best["is_ret"],
            }
            if prior_best
            else None
        ),
    }
    with open(ROOT / f"batch_{batch}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"batch {batch} done: {len(rows)} variants, {len(oos_pos)} OOS-profitable, "
        f"{len(both)} both-windows, {len(flukes)} OOS-only flukes, {time.time()-t0:.1f}s",
        flush=True,
    )
    if both:
        b0 = both[0]
        print(
            f"  top both-windows {b0['id']} {b0['family']} oos_profit={b0['oos_profit']} "
            f"is_ret={b0['is_ret']} trades={b0['oos_trades']}/{b0['is_trades']}",
            flush=True,
        )

    wf = run_walkforwards(D, batch)
    summary["walkforward"] = wf
    summary["seconds"] = round(time.time() - t0, 1)
    with open(ROOT / f"batch_{batch}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"all batch {batch} work finished in {time.time()-t0:.1f}s", flush=True)
    return summary


if __name__ == "__main__":
    run()
