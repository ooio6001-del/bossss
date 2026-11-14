#!/usr/bin/env python3
"""Batch 5 hunter: 10,000 NEW variants + walk-forward shortlist ranking.

Expands the token-washout gate family (Keltner, VWAP, WVF, SuperTrend entries
gated by a token-level hourly dump) with new lengths, dump thresholds, and
windows. Adds four fresh families: Elder Impulse, Connors RSI, MACD-histogram
turn, and hourly Double-N. After the 10,000, runs the Batch 4 multi-fold
walk-forward on each batch's top 20 both-window names plus H4-01464,
H3-05689, H3-05569, and H2-02108, and writes robust_leaderboard.csv of names
that pass. Walk-forward work is not part of the 10,000.
"""
from __future__ import annotations

import json
import re
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

import hunt as H
import hunt_batch2 as H2
import hunt_batch3 as H3
import hunt_batch4 as H4
import wf_lib as WF

DATA_DIR = H.DATA_DIR
BATCH_SIZE = 10_000
NONE = H.NONE

# New exits — none of these tuples were in batches 1–4
EXITS = [
    (0.18, 0.20, NONE, 0.0, 60, NONE, 1.0),
    (0.55, 0.24, 0.35, 0.16, 180, NONE, 1.0),
    (0.80, 0.27, NONE, 0.0, 252, NONE, 1.0),
    (NONE, 0.32, 0.55, 0.24, 432, NONE, 1.0),
    (1.20, 0.19, 0.38, 0.15, 336, NONE, 1.0),
]
# New diversified sizing — none of these (frac, maxpos) pairs were used before
SIZING = [(0.035, 28), (0.075, 13), (0.13, 8), (0.17, 5), (0.21, 4)]
BALANCES = np.array([45.0, 90.0, 160.0, 200.0, 300.0], dtype=np.float64)

NAMED_IDS = ["H4-01464", "H3-05689", "H3-05569", "H2-02108"]


def _pct(x):
    return f"{x * 100:g}%"


KEL_RE = re.compile(r"^kel(\d+)_(\d+(?:\.\d+)?)(s)?$")


def parse_kel(key):
    m = KEL_RE.match(key)
    if not m:
        raise ValueError(f"bad keltner key: {key}")
    return int(m.group(1)), float(m.group(2)), bool(m.group(3))


def _kel_txt(key):
    Lbar, k, squeeze = parse_kel(key)
    s = f"{Lbar}h Keltner (EMA+{k:g}×ATR)"
    return s + (" squeeze break" if squeeze else " break")


def _entry_label(key):
    if key.startswith("vwap"):
        return f"{key[4:]}h VWAP reclaim"
    if key.startswith("kel"):
        return _kel_txt(key)
    if key.startswith("wvf"):
        per, thr = key[3:].split("_", 1)
        return f"Williams VIX Fix {per}h at the {thr}rd percentile".replace("82rd", "82nd")
    if key.startswith("st"):
        per, m = key[2:].split("_", 1)
        return f"SuperTrend {per}h×{m} flip to long"
    return key


def _gate_label(key):
    _, pct, win = key.split("_")
    return f"the token dumped {pct}%+ in an hour within the last {win}h"


def entry_text(fam, p):
    if fam == "washout_gate":
        return p[-1] if isinstance(p[-1], str) else (
            f"Washout-gate: {_entry_label(p[0])} only if {_gate_label(p[1])}"
        )
    if fam == "elder_impulse":
        ema, f, s, sig, mode = int(p[0]), int(p[1]), int(p[2]), int(p[3]), int(p[4])
        base = f"EMA {ema} rising and MACD {f}/{s}/{sig} histogram rising"
        if mode == 0:
            return f"Buy the first Elder Impulse green bar ({base}) after a non-green bar"
        return f"Buy when the MACD {f}/{s}/{sig} histogram turns up while EMA {ema} is rising"
    if fam == "connors_rsi":
        return (
            f"Buy when RSI({int(p[0])}) crosses below {p[1]:g} while price is above "
            f"its {int(p[2])}h SMA (Connors-style dip in an uptrend)"
        )
    if fam == "macd_hist_turn":
        f, s, sig, mode = int(p[0]), int(p[1]), int(p[2]), int(p[3])
        macd = f"MACD {f}/{s}/{sig} histogram"
        if mode == 0:
            return f"Buy when the {macd} turns up from a trough"
        if mode == 1:
            return f"Buy when the {macd} turns up from a trough while still below zero"
        if mode == 2:
            return f"Buy when the {macd} crosses up through zero"
        return f"Buy when the {macd} turns up from a trough while EMA 13 is rising"
    if fam == "double_n":
        return (
            f"Buy the first hour of a {int(p[0])}-bar closing low while price is above "
            f"its {int(p[1])}h SMA (Double-N, hourly)"
        )
    return f"{fam} {p}"


def entries():
    """80 new entries: 48 token-washout gates + 32 fresh (4 families × 8)."""
    Elist = []
    wash = [
        ("vwap10", "dump_10_16"),
        ("vwap10", "dump_14_20"),
        ("vwap10", "dump_25_30"),
        ("vwap14", "dump_8_10"),
        ("vwap14", "dump_18_30"),
        ("vwap14", "dump_14_48"),
        ("vwap15", "dump_10_48"),
        ("vwap15", "dump_18_10"),
        ("vwap18", "dump_14_16"),
        ("vwap18", "dump_25_20"),
        ("vwap10", "dump_8_48"),
        ("vwap18", "dump_10_30"),
        ("kel10_1.8", "dump_10_16"),
        ("kel10_1.8s", "dump_14_20"),
        ("kel10_2.2s", "dump_18_10"),
        ("kel10_2.8", "dump_18_48"),
        ("kel16_2.2", "dump_8_10"),
        ("kel16_2.2s", "dump_18_30"),
        ("kel16_2.8", "dump_10_48"),
        ("kel16_1.8", "dump_25_16"),
        ("kel16_2.2", "dump_14_10"),
        ("kel18_1.8", "dump_14_16"),
        ("kel18_2.2s", "dump_25_20"),
        ("kel18_2.8", "dump_8_30"),
        ("kel18_2.2", "dump_10_48"),
        ("kel24_1.8s", "dump_10_16"),
        ("kel24_2.2", "dump_18_20"),
        ("kel24_2.8s", "dump_14_48"),
        ("wvf18_82", "dump_10_16"),
        ("wvf18_86", "dump_14_30"),
        ("wvf18_82", "dump_25_20"),
        ("wvf20_82", "dump_8_20"),
        ("wvf20_86", "dump_18_48"),
        ("wvf20_86", "dump_10_16"),
        ("wvf16_84", "dump_10_10"),
        ("wvf16_86", "dump_18_20"),
        ("wvf24_82", "dump_14_16"),
        ("wvf24_86", "dump_8_30"),
        ("st8_2.5", "dump_10_16"),
        ("st8_3.5", "dump_14_20"),
        ("st8_3.5", "dump_18_10"),
        ("st8_2.5", "dump_8_48"),
        ("st12_2.5", "dump_8_10"),
        ("st12_3.5", "dump_18_30"),
        ("st12_2.5", "dump_25_16"),
        ("st12_3.5", "dump_10_20"),
        ("st16_2.5", "dump_10_48"),
        ("st16_3.5", "dump_14_30"),
    ]
    for ek, gk in wash:
        desc = f"Washout-gate: {_entry_label(ek)} only if {_gate_label(gk)}"
        Elist.append(("washout_gate", (ek, gk, desc)))

    for p in (
        (13, 12, 26, 9, 0),
        (13, 12, 26, 9, 1),
        (8, 8, 21, 5, 0),
        (8, 8, 21, 5, 1),
        (21, 12, 26, 9, 0),
        (21, 5, 35, 5, 1),
        (13, 5, 35, 5, 0),
        (8, 12, 26, 9, 1),
    ):
        Elist.append(("elder_impulse", p))

    for p in (
        (2, 5, 72),
        (2, 10, 72),
        (2, 5, 168),
        (2, 10, 168),
        (3, 10, 168),
        (3, 15, 336),
        (2, 8, 336),
        (2, 10, 480),
    ):
        Elist.append(("connors_rsi", p))

    for p in (
        (12, 26, 9, 0),
        (12, 26, 9, 1),
        (12, 26, 9, 2),
        (12, 26, 9, 3),
        (8, 21, 5, 1),
        (8, 21, 5, 3),
        (5, 35, 5, 1),
        (16, 32, 9, 3),
    ):
        Elist.append(("macd_hist_turn", p))

    for p in (
        (7, 48),
        (7, 168),
        (12, 72),
        (12, 168),
        (24, 168),
        (24, 336),
        (5, 72),
        (16, 240),
    ):
        Elist.append(("double_n", p))

    if len(Elist) != 80:
        raise RuntimeError(f"expected 80 entries, got {len(Elist)}")
    return Elist


def wilder(x, n):
    """Wilder smoothing, seeded with an n-bar SMA. x is (T, N)."""
    x = np.asarray(x, dtype=np.float64)
    T, N = x.shape
    out = np.full((T, N), np.nan, dtype=np.float64)
    if T < n:
        return out
    out[n - 1] = np.mean(x[:n], axis=0)
    alpha = 1.0 / n
    for t in range(n, T):
        prev = out[t - 1]
        xt = x[t]
        # carry NaN columns forward as NaN until a valid seed exists
        out[t] = np.where(np.isfinite(prev), prev + alpha * (xt - prev), xt)
    return out


def rsi_grid(C, n):
    d = np.zeros_like(C, dtype=np.float64)
    d[1:] = C[1:] - C[:-1]
    gain = np.clip(d, 0.0, None)
    loss = np.clip(-d, 0.0, None)
    ag = wilder(gain, n)
    al = wilder(loss, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        rs = ag / al
        out = 100.0 - 100.0 / (1.0 + rs)
        out = np.where(al == 0.0, np.where(ag == 0.0, np.nan, 100.0), out)
    return out


def macd_hist(C, fast, slow, sig):
    ef = H3.ema(C, fast)
    es = H3.ema(C, slow)
    line = ef - es
    signal = H3.ema(line, sig)
    return line - signal


def rolling_min(C, n):
    T, N = C.shape
    out = np.full((T, N), np.nan, dtype=np.float64)
    for t in range(n - 1, T):
        out[t] = np.min(C[t - n + 1 : t + 1], axis=0)
    return out


@njit(cache=True, parallel=True)
def sig_elder(ema, hist, first, grp, mode, t_hi):
    T, N = ema.shape
    out = np.zeros((T, N), dtype=np.bool_)
    for j in prange(N):
        if grp[j] == 2 or first[j] < 0:
            continue
        f0 = int(first[j])
        prev_green = False
        start = max(f0 + 3, 3)
        for t in range(start, t_hi):
            ev, eh = ema[t, j], hist[t, j]
            ev1, eh1, eh2 = ema[t - 1, j], hist[t - 1, j], hist[t - 2, j]
            if not (ev == ev and ev1 == ev1 and eh == eh and eh1 == eh1 and eh2 == eh2):
                prev_green = False
                continue
            ema_up = ev > ev1
            hist_up = eh > eh1
            green = ema_up and hist_up
            if mode == 0:
                if green and not prev_green:
                    out[t, j] = True
            else:
                if ema_up and eh > eh1 and eh1 <= eh2:
                    out[t, j] = True
            prev_green = green
    return out


@njit(cache=True, parallel=True)
def sig_connors(C, rsi, sma, first, grp, thr, t_hi):
    T, N = C.shape
    out = np.zeros((T, N), dtype=np.bool_)
    for j in prange(N):
        if grp[j] == 2 or first[j] < 0:
            continue
        f0 = int(first[j])
        start = max(f0 + 2, 2)
        for t in range(start, t_hi):
            r, r1, s, c = rsi[t, j], rsi[t - 1, j], sma[t, j], C[t, j]
            if not (r == r and r1 == r1 and s == s and c == c and c > 0 and s > 0):
                continue
            if r < thr and r1 >= thr and c > s:
                out[t, j] = True
    return out


@njit(cache=True, parallel=True)
def sig_hist_turn(hist, ema, first, grp, mode, t_hi):
    T, N = hist.shape
    out = np.zeros((T, N), dtype=np.bool_)
    for j in prange(N):
        if grp[j] == 2 or first[j] < 0:
            continue
        f0 = int(first[j])
        start = max(f0 + 3, 3)
        for t in range(start, t_hi):
            h, h1, h2 = hist[t, j], hist[t - 1, j], hist[t - 2, j]
            if not (h == h and h1 == h1 and h2 == h2):
                continue
            if mode == 2:
                if h1 <= 0.0 and h > 0.0:
                    out[t, j] = True
                continue
            trough = h > h1 and h1 <= h2
            if not trough:
                continue
            if mode == 0:
                out[t, j] = True
            elif mode == 1:
                if h < 0.0:
                    out[t, j] = True
            else:
                ev, ev1 = ema[t, j], ema[t - 1, j]
                if ev == ev and ev1 == ev1 and ev > ev1:
                    out[t, j] = True
    return out


@njit(cache=True, parallel=True)
def sig_double_n(C, rmin, sma, first, grp, t_hi):
    T, N = C.shape
    out = np.zeros((T, N), dtype=np.bool_)
    for j in prange(N):
        if grp[j] == 2 or first[j] < 0:
            continue
        f0 = int(first[j])
        start = max(f0 + 2, 2)
        prev = False
        for t in range(start, t_hi):
            c, mn, s = C[t, j], rmin[t, j], sma[t, j]
            ok = False
            if c == c and mn == mn and s == s and c > 0 and s > 0 and c <= mn and c > s:
                ok = True
            if ok and not prev:
                out[t, j] = True
            prev = ok
    return out


def get_entry_signal(key, D, cache):
    C, V, H_, L_ = D["C"], D["V"], D["H"], D["L"]
    first, grp = D["first"], D["grp"]
    t_hi = D["T"] - 1
    av24 = cache["av24"]
    ck = ("entry5", key)
    if ck in cache:
        return cache[ck]
    if key.startswith("vwap"):
        Lbar = int(key[4:])
        vwap = cache.setdefault(("vwap", Lbar), H.rolling_vwap(C, V, Lbar))
        sig = H.sig_vwap_reclaim(C, V, first, grp, vwap, av24, Lbar, 0.0, t_hi)
    elif key.startswith("kel"):
        Lbar, k, squeeze = parse_kel(key)
        mid = cache.setdefault(("ema", Lbar), H3.ema(C, Lbar))
        atr = cache.setdefault(("atr", Lbar), H.rolling_mean(H3.true_range(H_, L_, C), Lbar))
        atr_ma = cache.setdefault(("atrma48",), H.rolling_mean(H3.true_range(H_, L_, C), 48))
        upper = mid + k * atr
        sig = H3.sig_keltner(C, first, grp, mid, upper, atr, atr_ma, 1 if squeeze else 0, t_hi)
    elif key.startswith("wvf"):
        per, thr = key[3:].split("_", 1)
        per, thr = int(per), float(thr) / 100.0
        sig, _ext = H4.sig_wvf(H_, L_, C, first, grp, per, thr, t_hi)
    elif key.startswith("st"):
        per, m = key[2:].split("_", 1)
        per, m = int(per), float(m)
        atr = cache.setdefault(("atr", per), H.rolling_mean(H3.true_range(H_, L_, C), per))
        sig = H4.supertrend_sig(H_, L_, C, atr, m, first, grp, t_hi, None, 0.0)
    else:
        raise KeyError(key)
    cache[ck] = sig
    return sig


def get_gate(key, D, cache):
    C = D["C"]
    first, grp = D["first"], D["grp"]
    t_hi = D["T"] - 1
    ck = ("gate5", key)
    if ck in cache:
        return cache[ck]
    if not key.startswith("dump_"):
        raise KeyError(key)
    _, pct, win = key.split("_")
    dmp, win = int(pct) / 100.0, int(win)
    dump = cache.setdefault(("hourly_dump", dmp), H4.hourly_dump(C, first, grp, dmp, t_hi))
    sig = cache.setdefault(("recent_dump", dmp, win), H3.recent_true(dump, win))
    cache[ck] = sig
    return sig


def make_signal(fam, p, D, cache):
    C, V = D["C"], D["V"]
    first, grp = D["first"], D["grp"]
    t_hi = D["T"] - 1
    if fam == "washout_gate":
        entry, gate, _desc = p[0], p[1], p[2]
        return get_entry_signal(entry, D, cache) & get_gate(gate, D, cache)
    if fam == "elder_impulse":
        ema_n, f, s, sigl, mode = int(p[0]), int(p[1]), int(p[2]), int(p[3]), int(p[4])
        ema = cache.setdefault(("ema", ema_n), H3.ema(C, ema_n))
        hist = cache.setdefault(("macd_hist", f, s, sigl), macd_hist(C, f, s, sigl))
        return sig_elder(ema, hist, first, grp, mode, t_hi)
    if fam == "connors_rsi":
        rn, thr, sl = int(p[0]), float(p[1]), int(p[2])
        rsi = cache.setdefault(("rsi", rn), rsi_grid(C, rn))
        sma = cache.setdefault(("sma", sl), H.rolling_mean(C, sl))
        return sig_connors(C, rsi, sma, first, grp, thr, t_hi)
    if fam == "macd_hist_turn":
        f, s, sigl, mode = int(p[0]), int(p[1]), int(p[2]), int(p[3])
        hist = cache.setdefault(("macd_hist", f, s, sigl), macd_hist(C, f, s, sigl))
        ema13 = cache.setdefault(("ema", 13), H3.ema(C, 13))
        return sig_hist_turn(hist, ema13, first, grp, mode, t_hi)
    if fam == "double_n":
        n, sl = int(p[0]), int(p[1])
        rmin = cache.setdefault(("rmin", n), rolling_min(C, n))
        sma = cache.setdefault(("sma", sl), H.rolling_mean(C, sl))
        return sig_double_n(C, rmin, sma, first, grp, t_hi)
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


def run_shortlist_walkforward(D, batch_tag):
    print("walk-forward shortlist: top 20 both-window per batch + named leaders", flush=True)
    shortlist, _ = WF.load_shortlist(n_per_batch=20, extra_ids=NAMED_IDS, max_batch=5)
    print(f"  shortlist size {len(shortlist)}", flush=True)
    # Replay sanity on named configs at the original split
    split = int(D.get("split", int(D["T"] * 0.60)))
    caches = {}
    for i in NAMED_IDS + ["H1-00094"]:
        path = None
        for b in range(1, 6):
            p = ROOT / f"batch_{b}.csv"
            if not p.exists():
                continue
            for r in H.load_csv(p):
                if r["id"] == i:
                    path = r
                    break
            if path:
                break
        if not path:
            print(f"  WARN missing {i} for replay check", flush=True)
            continue
        sig = WF.signal_from_row(path, D, caches)
        e = np.asarray(json.loads(path["exit_params"]), float)
        tr = E.gen_trades(sig, e, D["O"], D["H"], D["L"], D["C"], D["last"], D["supply"], False)
        tok, ti, to, pin, pout, rs = [np.asarray(x) for x in tr]
        frac, maxpos = WF.parse_sizing(path["sizing"])
        bal = float(path["balance"])
        vol = D["V"][np.maximum(ti.astype(np.int64) - 1, 0), tok.astype(np.int64)]
        oos = WF._sim_window(
            tok.astype(np.int64), ti.astype(np.int64), to.astype(np.int64),
            pin, pout, D, vol, split, D["T"], frac, maxpos, bal,
        )
        print(
            f"  replay {i}: published oos_profit={path['oos_profit']} "
            f"replay={round(float(oos[0]-bal),4)} trades_pub={path['oos_trades']} "
            f"replay_tr={int(oos[4])}",
            flush=True,
        )

    folds, summaries = WF.walk_forward_shortlist(D, shortlist, original_split=split)
    # Rank the full shortlist by rolling robustness; keep passers for the robust board
    ranked = sorted(summaries, key=WF.robustness_key, reverse=True)
    WF.write_fold_csv(ROOT / "walkforward_shortlist_folds.csv", folds)
    # drop heavy _summary before writing the csv
    slim = [{k: v for k, v in r.items() if k != "_summary"} for r in ranked]
    WF.write_summary_csv(ROOT / "walkforward_shortlist.csv", slim)
    passed = WF.write_robust_leaderboard(ranked)
    # JSON summary
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

    named = [r for r in ranked if r["id"] in set(NAMED_IDS) or r["id"] == "H1-00094"]
    b5 = [r for r in ranked if str(r["batch"]) == "5" or str(r["id"]).startswith("H5-")]
    payload = {
        "pass_rule": WF.pass_rule_text(),
        "n_shortlist": len(ranked),
        "n_passed": len(passed),
        "named": [slim_sum(r) for r in named],
        "batch5_top20_by_rolling": [slim_sum(r) for r in b5],
        "robust_leaderboard": [slim_sum(r) for r in passed],
        "all_shortlist_by_rolling": [slim_sum(r) for r in ranked],
    }
    with open(ROOT / "walkforward_shortlist.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(
        f"walk-forward done: shortlist={len(ranked)} passed={len(passed)} "
        f"wrote robust_leaderboard.csv",
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

    prior = [r for r in all_rows if int(r["batch"]) < batch]
    prior_both = [
        r for r in prior
        if r.get("both_windows") == "true" or (float(r["oos_profit"]) > 0 and float(r["is_ret"]) > 0)
    ]
    prior_both.sort(key=lambda r: (float(r["oos_profit"]), float(r.get("is_profit") or 0)), reverse=True)
    prior_best = prior_both[0] if prior_both else (prior[0] if prior else None)

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
                "both_windows": prior_best.get("both_windows", ""),
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

    # Walk-forward after the 10,000 so batch 5 top-20 are on disk
    wf = run_shortlist_walkforward(D, batch)
    summary["walkforward"] = {
        "n_shortlist": wf["n_shortlist"],
        "n_passed": wf["n_passed"],
        "pass_rule": wf["pass_rule"],
        "named": wf["named"],
        "batch5_top20_by_rolling": wf["batch5_top20_by_rolling"],
        "robust_leaderboard": wf["robust_leaderboard"],
    }
    summary["seconds"] = round(time.time() - t0, 1)
    with open(ROOT / f"batch_{batch}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"all batch {batch} work finished in {time.time()-t0:.1f}s", flush=True)
    return summary


if __name__ == "__main__":
    run()
