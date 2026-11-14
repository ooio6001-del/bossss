#!/usr/bin/env python3
"""Shared rolling walk-forward helpers for batch 5 onward.

Same fold design as the Batch 4 retest of H1-00094: the original 60/40 split,
anchored 40/50/60/70/80% cuts, calendar month-end splits, and three rolling
walk-forward schedules. Not counted in a batch's 10,000 variants.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from solstrats import costs as K
from solstrats import engine as E

import hunt as H
import hunt_batch2 as H2
import hunt_batch3 as H3
import hunt_batch4 as H4

CHI = ZoneInfo("America/Chicago")
NONE = H.NONE

# Pass bar for robust_leaderboard.csv. Documented in ideas.md and the report.
PASS_MIN_OOS_SHARE = 0.50
PASS_MIN_TRADED_FOLDS = 10

WF_FOLD_FIELDS = [
    "id",
    "fold",
    "balance",
    "is_lo",
    "is_hi",
    "is_start",
    "is_end",
    "is_end_bal",
    "is_profit",
    "is_ret",
    "is_win",
    "is_trades",
    "is_dd",
    "is_sharpe",
    "is_pf",
    "oos_lo",
    "oos_hi",
    "oos_start",
    "oos_end",
    "oos_end_bal",
    "oos_profit",
    "oos_ret",
    "oos_win",
    "oos_trades",
    "oos_dd",
    "oos_sharpe",
    "oos_pf",
    "both_windows",
]

ROBUST_FIELDS = [
    "id",
    "batch",
    "family",
    "entry",
    "exit",
    "sizing",
    "balance",
    "passed",
    "wf_n_folds",
    "wf_oos_profitable_pct",
    "wf_both_windows_pct",
    "wf_median_oos_ret",
    "wf_mean_oos_ret",
    "wf_min_oos_ret",
    "wf_folds_with_oos_trades",
    "wf_roll_n",
    "wf_roll_oos_profitable_pct",
    "wf_roll_median_oos_ret",
    "wf_roll_mean_oos_ret",
    "wf_full_ret",
    "wf_full_profit",
    "wf_full_trades",
    "wf_full_dd",
    "orig_oos_profit",
    "orig_oos_ret",
    "orig_is_ret",
    "orig_oos_trades",
    "orig_is_trades",
    "orig_full_ret",
    "both_windows",
    "params",
    "exit_params",
]


def _chi(ts):
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).astimezone(CHI).strftime(
        "%Y-%m-%d %I:%M %p %Z"
    )


def parse_sizing(text):
    m = re.search(r"([\d.]+)% of balance per trade, up to (\d+) open", str(text))
    if not m:
        raise ValueError(f"cannot parse sizing: {text!r}")
    return float(m.group(1)) / 100.0, int(m.group(2))


def _sim_window(tok, ti, to, pin, pout, D, vol_entry, lo, hi, frac, maxpos, bal):
    empty = np.zeros(0)
    return E.simulate(
        tok,
        ti,
        to,
        pin,
        pout,
        D["venue"],
        vol_entry,
        D["solpx"],
        lo,
        hi,
        bal,
        frac,
        maxpos,
        K.PCT_COST,
        K.FIXED_SOL,
        K.RENT_SOL,
        K.RENT_UNRECOVERED,
        K.IMPACT_K,
        K.FILL_CAP,
        empty,
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


def build_fold_specs(T, times, original_split=None):
    """Return (name, is_lo, is_hi, oos_hi) using the Batch 4 fold design."""
    if original_split is None:
        original_split = int(T * 0.60)
    specs = [("anchored_60pct_original_split", 0, int(original_split), T)]
    for f in (0.40, 0.50, 0.60, 0.70, 0.80):
        mid = int(T * f)
        if mid < 24 or T - mid < 24:
            continue
        if mid == int(original_split):
            continue
        specs.append((f"anchored_{int(f * 100)}pct", 0, mid, T))

    month_end_bars = {}
    for t in range(T):
        dt = datetime.fromtimestamp(float(times[t]), tz=timezone.utc)
        key = (dt.year, dt.month)
        month_end_bars[key] = t
    for (y, m), bar in sorted(month_end_bars.items()):
        mid = bar + 1
        if mid < 24 * 30 or T - mid < 24 * 14:
            continue
        specs.append((f"calendar_split_after_{y}-{m:02d}", 0, mid, T))

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
            specs.append((f"{tag}_{k:02d}", is_lo, is_hi, oos_hi))
            start += step
    return specs


def _subset(folds, prefix):
    if prefix == "anchored":
        rows = [r for r in folds if r["fold"].startswith("anchored")]
    elif prefix == "calendar":
        rows = [r for r in folds if r["fold"].startswith("calendar")]
    elif prefix == "roll":
        rows = [r for r in folds if r["fold"].startswith("roll")]
    else:
        rows = list(folds)
    rets = [float(r["oos_ret"]) for r in rows]
    oos_pos = sum(1 for r in rows if float(r["oos_profit"]) > 0)
    both = sum(1 for r in rows if r["both_windows"] == "true")
    traded = sum(1 for r in rows if int(r["oos_trades"]) > 0)
    return {
        "n": len(rows),
        "oos_profitable": oos_pos,
        "both_windows": both,
        "folds_with_oos_trades": traded,
        "oos_profitable_pct": round(100.0 * oos_pos / len(rows), 2) if rows else None,
        "both_windows_pct": round(100.0 * both / len(rows), 2) if rows else None,
        "oos_ret_mean": round(float(np.mean(rets)), 4) if rets else None,
        "oos_ret_median": round(float(np.median(rets)), 4) if rets else None,
        "oos_ret_min": round(float(np.min(rets)), 4) if rets else None,
        "oos_ret_max": round(float(np.max(rets)), 4) if rets else None,
    }


def summarize_folds(folds, full_m, bal, n_raw=None):
    full = {
        "end_bal": round(float(full_m[0]), 4),
        "profit": round(float(full_m[0] - bal), 4),
        "ret": round(float(full_m[1]), 4),
        "win": round(float(full_m[13]), 4),
        "trades": int(full_m[4]),
        "dd": round(float(full_m[2]), 4),
        "sharpe": round(float(full_m[3]), 4),
        "pf": round(float(full_m[12]), 4),
        "fees": round(float(full_m[11]), 4),
    }
    all_ = _subset(folds, "all")
    roll = _subset(folds, "roll")
    passed = passes_walk_forward(all_, roll, full)
    out = {
        "n_folds": len(folds),
        "n_raw_trades": n_raw,
        "aggregates": {
            "all_folds": all_,
            "anchored": _subset(folds, "anchored"),
            "calendar": _subset(folds, "calendar"),
            "rolling": roll,
        },
        "full_period": full,
        "passed": passed,
        "pass_rule": pass_rule_text(),
    }
    return out


def pass_rule_text():
    return (
        "Pass if full-period return > 0, median OOS return across all folds > 0, "
        f"at least {PASS_MIN_OOS_SHARE:.0%} of all folds are OOS-profitable, "
        f"and at least {PASS_MIN_TRADED_FOLDS} folds have one or more OOS trades."
    )


def passes_walk_forward(all_agg, roll_agg, full):
    if not all_agg or all_agg["n"] == 0:
        return False
    med = all_agg["oos_ret_median"]
    share = (all_agg["oos_profitable"] / all_agg["n"]) if all_agg["n"] else 0.0
    traded = all_agg["folds_with_oos_trades"]
    return (
        float(full["ret"]) > 0
        and med is not None
        and float(med) > 0
        and share >= PASS_MIN_OOS_SHARE
        and int(traded) >= PASS_MIN_TRADED_FOLDS
    )


def walk_forward_trades(D, tok, ti, to, pin, pout, frac, maxpos, bal, original_split=None, end_bar=None):
    """Run the standard fold set on an already-built trade list.

    If end_bar is set, folds and the full-period sim stop there (used to keep
    a later holdout out of walk-forward).
    """
    tok = np.asarray(tok, dtype=np.int64)
    ti = np.asarray(ti, dtype=np.int64)
    to = np.asarray(to, dtype=np.int64)
    pin = np.asarray(pin, dtype=np.float64)
    pout = np.asarray(pout, dtype=np.float64)
    vol_entry = D["V"][np.maximum(ti - 1, 0), tok] if tok.size else np.zeros(0)
    T = int(D["T"] if end_bar is None else end_bar)
    times = D["times"][:T]
    if original_split is None:
        original_split = int(D.get("split", int(T * 0.60)))
    if original_split >= T:
        original_split = int(T * 0.60)

    specs = build_fold_specs(T, times, original_split=original_split)

    def run_fold(name, is_lo, is_hi, oos_hi):
        is_m = _sim_window(tok, ti, to, pin, pout, D, vol_entry, is_lo, is_hi, frac, maxpos, bal)
        oos_m = _sim_window(tok, ti, to, pin, pout, D, vol_entry, is_hi, oos_hi, frac, maxpos, bal)
        return _fold_row(name, is_lo, is_hi, oos_hi, times, is_m, oos_m, bal)

    folds = [run_fold(*spec) for spec in specs]
    full = _sim_window(tok, ti, to, pin, pout, D, vol_entry, 0, T, frac, maxpos, bal)
    summary = summarize_folds(folds, full, bal, n_raw=int(tok.size))
    return folds, summary


def _as_entry_tuple(raw, fam):
    if isinstance(raw, dict):
        if fam == "confluence":
            return (raw["entry"], raw["gate"], raw.get("hod", 0), raw.get("desc", ""))
        if fam == "ensemble":
            comps = raw["comps"]
            comps = tuple(comps) if not isinstance(comps, tuple) else comps
            return (raw["mode"], comps, raw.get("desc", ""))
        if fam == "washout_gate":
            return (raw["entry"], raw["gate"], raw.get("desc", ""))
        raise KeyError(f"dict params for family {fam}")
    return tuple(raw)


def signal_from_row(row, D, caches):
    """Rebuild a signal for a saved CSV row from batches 1–5."""
    fam = row["family"]
    if fam == "moonshot" or str(row.get("batch")) == "sweep":
        b = "sweep"
    else:
        b = int(row["batch"])
    raw = json.loads(row["params"]) if isinstance(row.get("params"), str) else row["params"]
    if fam != "moonshot":
        p = _as_entry_tuple(raw, fam)
    cache = caches.setdefault(b, {})
    if "av24" not in cache:
        cache["av24"] = H.avg_vol(D["V"], 24)
    if "hod" not in cache:
        cache["hod"] = ((D["times"].astype(np.int64) % 86400) // 3600).astype(np.int64)
    if b == 1:
        return H.make_signal(fam, p, D, cache)
    if b == 2:
        return H2.make_signal(fam, p, D, cache)
    if b == 3:
        return H3.make_signal(fam, p, D, cache)
    if b == 4:
        return H4.make_signal(fam, p, D, cache)
    if fam == "moonshot" or str(row.get("batch")) == "sweep":
        rawp = json.loads(row["params"]) if isinstance(row["params"], str) else row["params"]
        pp = np.asarray(rawp, float)
        hot = D["hot"][24] if isinstance(D.get("hot"), dict) else D["hot"]
        return E.signals(
            5,
            pp,
            D["O"],
            D["H"],
            D["L"],
            D["C"],
            D["V"],
            D["first"],
            D["grp"],
            D["venue"],
            D["supply"],
            D["mint_auth"],
            D["freeze"],
            D["top10"],
            hot,
            D["theme"],
            0,
            D["T"] - 1,
        )
    if b == 5:
        import hunt_batch5 as H5

        return H5.make_signal(fam, p, D, cache)
    if b == 6:
        import hunt_batch6 as H6

        return H6.make_signal(fam, p, D, cache)
    if b == 7:
        import hunt_batch7 as H7

        return H7.make_signal(fam, p, D, cache)
    if b == 8:
        import hunt_batch8 as H8

        return H8.make_signal(fam, p, D, cache)
    raise KeyError(f"unknown batch {b} family {fam}")


def top_both_window(rows, n=20):
    both = [
        r
        for r in rows
        if str(r.get("both_windows", "")).lower() == "true"
        or (float(r["oos_profit"]) > 0 and float(r["is_ret"]) > 0)
    ]
    both.sort(key=lambda r: (float(r["oos_profit"]), float(r.get("is_profit") or r.get("is_ret") or 0)), reverse=True)
    return both[:n]


def load_shortlist(n_per_batch=20, extra_ids=None, max_batch=5):
    """Top-N both-window names from each existing batch, plus any named extras."""
    seen = set()
    out = []
    extra_ids = list(extra_ids or [])
    by_id = {}
    for b in range(1, max_batch + 1):
        path = ROOT / f"batch_{b}.csv"
        if not path.exists():
            continue
        rows = H.load_csv(path)
        for r in rows:
            by_id[r["id"]] = r
        for r in top_both_window(rows, n_per_batch):
            if r["id"] not in seen:
                seen.add(r["id"])
                rec = dict(r)
                rec["_source"] = f"batch{b}_top{n_per_batch}_both"
                out.append(rec)
    for i in extra_ids:
        if i in seen:
            # still tag that it was requested by name
            for r in out:
                if r["id"] == i:
                    r["_source"] = r.get("_source", "") + "+named"
            continue
        if i not in by_id:
            # search later batches already loaded; if missing, skip with a note
            continue
        rec = dict(by_id[i])
        rec["_source"] = "named"
        seen.add(i)
        out.append(rec)
    return out, by_id


def summary_to_row(src_row, summary):
    all_ = summary["aggregates"]["all_folds"]
    roll = summary["aggregates"]["rolling"]
    full = summary["full_period"]
    return {
        "id": src_row["id"],
        "batch": src_row["batch"],
        "family": src_row["family"],
        "entry": src_row["entry"],
        "exit": src_row["exit"],
        "sizing": src_row["sizing"],
        "balance": src_row["balance"],
        "passed": "true" if summary["passed"] else "false",
        "wf_n_folds": all_["n"],
        "wf_oos_profitable_pct": all_["oos_profitable_pct"],
        "wf_both_windows_pct": all_["both_windows_pct"],
        "wf_median_oos_ret": all_["oos_ret_median"],
        "wf_mean_oos_ret": all_["oos_ret_mean"],
        "wf_min_oos_ret": all_["oos_ret_min"],
        "wf_folds_with_oos_trades": all_["folds_with_oos_trades"],
        "wf_roll_n": roll["n"],
        "wf_roll_oos_profitable_pct": roll["oos_profitable_pct"],
        "wf_roll_median_oos_ret": roll["oos_ret_median"],
        "wf_roll_mean_oos_ret": roll["oos_ret_mean"],
        "wf_full_ret": full["ret"],
        "wf_full_profit": full["profit"],
        "wf_full_trades": full["trades"],
        "wf_full_dd": full["dd"],
        "orig_oos_profit": src_row.get("oos_profit", ""),
        "orig_oos_ret": src_row.get("oos_ret", ""),
        "orig_is_ret": src_row.get("is_ret", ""),
        "orig_oos_trades": src_row.get("oos_trades", ""),
        "orig_is_trades": src_row.get("is_trades", ""),
        "orig_full_ret": src_row.get("full_ret", ""),
        "both_windows": src_row.get("both_windows", ""),
        "params": src_row.get("params", ""),
        "exit_params": src_row.get("exit_params", ""),
        "_source": src_row.get("_source", ""),
    }


def robustness_key(r):
    """Sort key: rolling fold-profitable %, rolling median OOS, full-period return."""
    return (
        float(r.get("wf_roll_oos_profitable_pct") or 0),
        float(r.get("wf_roll_median_oos_ret") or 0),
        float(r.get("wf_full_ret") or 0),
    )


def walk_forward_shortlist(D, rows, original_split=None, end_bar=None):
    """Rebuild signals/trades as needed and walk-forward each saved config."""
    caches = {}
    sig_cache = {}
    trade_cache = {}
    all_folds = []
    summaries = []
    n = len(rows)
    for i, row in enumerate(rows, 1):
        sid = (
            str(row["batch"]),
            row["family"],
            row["params"],
        )
        eid = sid + (row["exit_params"],)
        if sid not in sig_cache:
            sig_cache[sid] = signal_from_row(row, D, caches)
        if eid not in trade_cache:
            e = np.asarray(json.loads(row["exit_params"]), float)
            one_shot = (
                str(row.get("one_shot", "")).lower() in ("true", "1")
                or row.get("family") == "moonshot"
            )
            tr = E.gen_trades(
                sig_cache[sid],
                e,
                D["O"],
                D["H"],
                D["L"],
                D["C"],
                D["last"],
                D["supply"],
                one_shot,
            )
            tok, ti, to, pin, pout, rs = [np.asarray(x) for x in tr]
            trade_cache[eid] = (
                tok.astype(np.int64),
                ti.astype(np.int64),
                to.astype(np.int64),
                pin,
                pout,
            )
        tok, ti, to, pin, pout = trade_cache[eid]
        frac, maxpos = parse_sizing(row["sizing"])
        bal = float(row["balance"])
        folds, summary = walk_forward_trades(
            D, tok, ti, to, pin, pout, frac, maxpos, bal,
            original_split=original_split, end_bar=end_bar,
        )
        for f in folds:
            f = dict(f)
            f["id"] = row["id"]
            all_folds.append(f)
        rec = summary_to_row(row, summary)
        rec["_summary"] = summary
        summaries.append(rec)
        if i == 1 or i % 10 == 0 or i == n:
            print(
                f"  walk-forward {i}/{n} {row['id']} "
                f"roll_oos%={rec['wf_roll_oos_profitable_pct']} "
                f"med={rec['wf_roll_median_oos_ret']} "
                f"full={rec['wf_full_ret']} "
                f"pass={rec['passed']}",
                flush=True,
            )
    return all_folds, summaries


def write_fold_csv(path, folds):
    path = Path(path)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=WF_FOLD_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in folds:
            w.writerow(r)


def write_summary_csv(path, rows, fields=None):
    fields = fields or [c for c in ROBUST_FIELDS]
    path = Path(path)
    # Keep a source column in the full shortlist file only.
    extra = []
    if any("_source" in r for r in rows) and "_source" not in fields:
        extra = ["_source"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields + extra, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_robust_leaderboard(summaries, path=None):
    path = path or (ROOT / "robust_leaderboard.csv")
    passed = [r for r in summaries if str(r.get("passed", "")).lower() == "true"]
    passed.sort(key=robustness_key, reverse=True)
    write_summary_csv(path, passed, fields=ROBUST_FIELDS)
    return passed
