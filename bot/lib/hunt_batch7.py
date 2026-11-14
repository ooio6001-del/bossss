#!/usr/bin/env python3
"""Batch 7 hunter: 10,000 NEW Double-N washout variants + a frozen holdout.

No new hourly bars arrived after batch 6, so the most recent 14 days already
in the file are reserved as a holdout. Batch 7 is ranked only on data before
that window. The robust top 10 (especially H6 Double-N washout) are then
scored on the holdout as the main credibility check. Walk-forward, when run,
also stops at the holdout start.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from solstrats import costs as K
from solstrats import engine as E
from solstrats.describe import exit_text, sizing_text

import hunt as H
import hunt_batch5 as H5
import hunt_batch6 as H6
import wf_lib as WF

DATA_DIR = H.DATA_DIR
BATCH_SIZE = 10_000
NONE = H.NONE
CHI = ZoneInfo("America/Chicago")
HOLDOUT_HOURS = 14 * 24  # 336 bars ≈ 2 weeks

# Nearby exits, not used in batches 1–6
EXITS = [
    (NONE, 0.29, 0.58, 0.18, 360, NONE, 1.0),
    (NONE, 0.33, 0.65, 0.22, 408, NONE, 1.0),
    (1.30, 0.25, NONE, 0.0, 264, NONE, 1.0),
    (1.60, 0.28, 0.50, 0.18, 312, NONE, 1.0),
    (NONE, 0.31, 0.70, 0.25, 456, NONE, 1.0),
]
SIZING = [(0.105, 9), (0.185, 5), (0.24, 4), (0.29, 3), (0.33, 3)]
BALANCES = np.array([65.0, 110.0, 190.0, 240.0, 350.0], dtype=np.float64)

HOLD_FIELDS = [
    "id", "batch", "family", "entry", "exit", "sizing", "balance",
    "selected_on", "hold_start", "hold_end", "hold_bars",
    "hold_profit", "hold_ret", "hold_trades", "hold_win", "hold_dd",
    "hold_sharpe", "hold_pf", "hold_profitable",
    "purged_oos_profit", "purged_oos_ret", "purged_oos_trades",
    "is_ret", "is_trades", "full_pre_ret", "orig_oos_profit", "orig_is_ret",
    "params", "exit_params",
]


def _chi(ts):
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).astimezone(CHI).strftime(
        "%Y-%m-%d %I:%M %p %Z"
    )


def _gate_label(key):
    _, pct, win = key.split("_")
    return f"the token dumped {pct}%+ in an hour within the last {win}h"


def _dn_label(n, sma):
    return f"first hour of a {n}-bar closing low above its {sma}h SMA"


def entry_text(fam, p):
    if fam == "washout_gate":
        return p[-1] if isinstance(p[-1], str) else str(p)
    return f"{fam} {p}"


def dn_pairs():
    """80 new Double-N washout pairs around the H6 leaders. No batch-6 repeats."""
    # Around dn8_120 + dump_16_16 (28)
    around_a = [
        (7, 120, "dump_16_16"), (9, 120, "dump_16_16"),
        (8, 108, "dump_16_16"), (8, 132, "dump_16_16"), (8, 144, "dump_16_16"),
        (8, 120, "dump_15_16"), (8, 120, "dump_17_16"), (8, 120, "dump_16_14"),
        (8, 120, "dump_16_18"), (8, 120, "dump_16_20"), (8, 120, "dump_15_14"),
        (8, 120, "dump_17_18"), (8, 120, "dump_14_16"), (8, 120, "dump_18_16"),
        (7, 108, "dump_15_14"), (7, 132, "dump_17_18"), (9, 108, "dump_17_14"),
        (9, 132, "dump_15_18"), (8, 108, "dump_15_18"), (8, 132, "dump_17_14"),
        (8, 144, "dump_15_16"), (8, 144, "dump_17_20"), (7, 120, "dump_16_20"),
        (9, 120, "dump_16_14"), (7, 144, "dump_16_16"), (9, 144, "dump_16_16"),
        (8, 108, "dump_17_20"), (9, 132, "dump_16_20"),
    ]
    # Around dn8_288 + dump_16_24 (26)
    around_b = [
        (7, 288, "dump_16_24"), (9, 288, "dump_16_24"),
        (8, 264, "dump_16_24"), (8, 312, "dump_16_24"), (8, 336, "dump_16_24"),
        (8, 288, "dump_15_24"), (8, 288, "dump_17_24"), (8, 288, "dump_16_20"),
        (8, 288, "dump_16_28"), (8, 288, "dump_16_22"), (8, 288, "dump_15_20"),
        (8, 288, "dump_17_28"), (7, 264, "dump_15_20"), (9, 312, "dump_17_28"),
        (8, 264, "dump_15_28"), (8, 312, "dump_17_20"), (7, 288, "dump_16_28"),
        (9, 288, "dump_16_20"), (8, 264, "dump_16_20"), (8, 312, "dump_16_28"),
        (8, 336, "dump_15_24"), (8, 336, "dump_17_24"), (7, 312, "dump_16_24"),
        (9, 264, "dump_16_24"), (8, 288, "dump_14_24"), (8, 288, "dump_18_28"),
    ]
    # Around dn10_288 + dump_16_36 (26)
    around_c = [
        (9, 288, "dump_16_36"), (11, 288, "dump_16_36"),
        (10, 264, "dump_16_36"), (10, 312, "dump_16_36"), (10, 336, "dump_16_36"),
        (10, 288, "dump_15_36"), (10, 288, "dump_17_36"), (10, 288, "dump_16_30"),
        (10, 288, "dump_16_32"), (10, 288, "dump_16_40"), (10, 288, "dump_15_30"),
        (10, 288, "dump_17_40"), (9, 264, "dump_15_32"), (11, 312, "dump_17_40"),
        (10, 264, "dump_17_30"), (10, 312, "dump_15_40"), (9, 288, "dump_16_30"),
        (11, 288, "dump_16_40"), (10, 264, "dump_16_30"), (10, 312, "dump_16_40"),
        (11, 264, "dump_16_36"), (9, 312, "dump_16_36"), (10, 288, "dump_14_36"),
        (10, 288, "dump_18_36"), (11, 336, "dump_15_32"), (9, 336, "dump_17_40"),
    ]
    pairs = around_a + around_b + around_c
    if len(pairs) != 80:
        raise RuntimeError(f"expected 80 pairs, got {len(pairs)}")
    if len(set(pairs)) != 80:
        raise RuntimeError("duplicate Double-N washout pairs")
    return pairs


def entries():
    Elist = []
    for n, sma, gk in dn_pairs():
        ek = f"dn{n}_{sma}"
        desc = f"Washout-gate: {_dn_label(n, sma)} only if {_gate_label(gk)}"
        Elist.append(("washout_gate", (ek, gk, desc)))
    return Elist


def make_signal(fam, p, D, cache):
    if fam != "washout_gate":
        raise KeyError(fam)
    sig = H6.get_entry_signal(p[0], D, cache)
    gate = H5.get_gate(p[1], D, cache)
    return sig & gate


def row_from(i, batch, fam, p, e, frac, maxpos, bal, oos, is_, full, hold, full_all):
    oos_profit = float(oos[0] - bal)
    is_ret = float(is_[1])
    hold_profit = float(hold[0] - bal)
    params = json.dumps({"entry": p[0], "gate": p[1], "desc": p[2]})
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


def pack(rows, n=12, min_trades=8, holdout=False):
    solid = [r for r in rows if int(r["oos_trades"]) >= min_trades]
    out = []
    for r in solid[:n]:
        rec = {
            "id": r["id"],
            "family": r["family"],
            "entry": r["entry"],
            "exit": r["exit"],
            "sizing": r["sizing"],
            "balance": r["balance"],
            "oos_profit": r["oos_profit"],
            "oos_ret": r["oos_ret"],
            "oos_trades": r["oos_trades"],
            "is_ret": r["is_ret"],
            "is_trades": r["is_trades"],
            "full_ret": r["full_ret"],
            "both_windows": r["both_windows"],
        }
        if holdout:
            rec.update({
                "holdout_profit": r.get("holdout_profit"),
                "holdout_ret": r.get("holdout_ret"),
                "holdout_trades": r.get("holdout_trades"),
                "holdout_dd": r.get("holdout_dd"),
            })
        out.append(rec)
    return out


def window_metrics(tok, ti, to, pin, pout, D, frac, maxpos, bal, lo, hi):
    vol = D["V"][np.maximum(ti - 1, 0), tok] if tok.size else np.zeros(0)
    m = WF._sim_window(tok, ti, to, pin, pout, D, vol, lo, hi, frac, maxpos, bal)
    return {
        "profit": round(float(m[0] - bal), 4),
        "ret": round(float(m[1]), 4),
        "trades": int(m[4]),
        "win": round(float(m[13]), 4),
        "dd": round(float(m[2]), 4),
        "sharpe": round(float(m[3]), 4),
        "pf": round(float(m[12]), 4),
        "end": round(float(m[0]), 4),
    }


def eval_row_windows(row, D, caches, purged_lo, hold_lo, T):
    sig = WF.signal_from_row(row, D, caches)
    e = np.asarray(json.loads(row["exit_params"]), float)
    tr = E.gen_trades(sig, e, D["O"], D["H"], D["L"], D["C"], D["last"], D["supply"], False)
    tok, ti, to, pin, pout, rs = [np.asarray(x) for x in tr]
    tok, ti, to = tok.astype(np.int64), ti.astype(np.int64), to.astype(np.int64)
    frac, maxpos = WF.parse_sizing(row["sizing"])
    bal = float(row["balance"])
    is_m = window_metrics(tok, ti, to, pin, pout, D, frac, maxpos, bal, 0, purged_lo and D.get("split", 0) or int(row.get("split", D["split"])))
    # always use D['split'] for IS
    is_m = window_metrics(tok, ti, to, pin, pout, D, frac, maxpos, bal, 0, int(D["split"]))
    purged = window_metrics(tok, ti, to, pin, pout, D, frac, maxpos, bal, int(D["split"]), hold_lo)
    hold = window_metrics(tok, ti, to, pin, pout, D, frac, maxpos, bal, hold_lo, T)
    pre = window_metrics(tok, ti, to, pin, pout, D, frac, maxpos, bal, 0, hold_lo)
    return is_m, purged, hold, pre, (tok, ti, to, pin, pout, frac, maxpos, bal)


def holdout_record(row, hold, purged, is_m, pre, hold_lo, T, times, selected_on):
    return {
        "id": row["id"],
        "batch": row.get("batch", ""),
        "family": row.get("family", ""),
        "entry": row.get("entry", ""),
        "exit": row.get("exit", ""),
        "sizing": row.get("sizing", ""),
        "balance": row.get("balance", ""),
        "selected_on": selected_on,
        "hold_start": _chi(times[hold_lo]),
        "hold_end": _chi(times[T - 1]),
        "hold_bars": int(T - hold_lo),
        "hold_profit": hold["profit"],
        "hold_ret": hold["ret"],
        "hold_trades": hold["trades"],
        "hold_win": hold["win"],
        "hold_dd": hold["dd"],
        "hold_sharpe": hold["sharpe"],
        "hold_pf": hold["pf"],
        "hold_profitable": "true" if hold["profit"] > 0 else "false",
        "purged_oos_profit": purged["profit"],
        "purged_oos_ret": purged["ret"],
        "purged_oos_trades": purged["trades"],
        "is_ret": is_m["ret"],
        "is_trades": is_m["trades"],
        "full_pre_ret": pre["ret"],
        "orig_oos_profit": row.get("oos_profit", ""),
        "orig_is_ret": row.get("is_ret", ""),
        "params": row.get("params", ""),
        "exit_params": row.get("exit_params", ""),
    }


def write_hold_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HOLD_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_extended_csv(path, rows):
    fields = list(H.FIELDS) + [
        "holdout_end", "holdout_profit", "holdout_ret", "holdout_trades",
        "holdout_win", "holdout_dd", "holdout_sharpe", "holdout_pf",
        "full_all_ret", "full_all_trades",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def standard_rows(rows):
    return [{k: r[k] for k in H.FIELDS} for r in rows]


def run():
    ROOT.mkdir(parents=True, exist_ok=True)
    batch = H.next_batch_number()
    t0 = time.time()
    print(f"batch {batch}: loading data", flush=True)
    D = H.prep(DATA_DIR)
    T, N = D["T"], D["N"]
    split = int(D["split"])
    times = D["times"]
    if T - HOLDOUT_HOURS <= split + 24:
        raise RuntimeError("holdout would leave too little pre-holdout OOS")
    hold_lo = T - HOLDOUT_HOURS
    print(
        f"data {T} hours x {N} tokens; IS 0-{split}; "
        f"purged OOS {split}-{hold_lo}; holdout {hold_lo}-{T} "
        f"({_chi(times[hold_lo])} -> {_chi(times[T-1])})",
        flush=True,
    )
    if T > 4297:
        print(f"  NOTE: {T - 4297} bars arrived after batch 6 and are inside the holdout", flush=True)
    else:
        print(
            "  NOTE: no new bars since batch 6; holdout is the last 14 days already "
            "inside prior batches' OOS. Batch 7 is ranked without it.",
            flush=True,
        )

    # refuse exact batch-6 repeats
    prior = set()
    b6 = ROOT / "batch_6.csv"
    if b6.exists():
        for r in H.load_csv(b6):
            if r["family"] == "washout_gate":
                p = json.loads(r["params"])
                prior.add((p["entry"], p["gate"]))
    new_pairs = [(f"dn{n}_{sma}", gk) for n, sma, gk in dn_pairs()]
    overlap = [x for x in new_pairs if x in prior]
    if overlap:
        raise RuntimeError(f"repeats of batch-6 pairs: {overlap[:5]}")

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
    windows = (
        ("is", 0, split),
        ("oos", split, hold_lo),          # purged OOS used for ranking
        ("full", 0, hold_lo),             # full pre-holdout
        ("hold", hold_lo, T),             # holdout, not used for ranking
        ("full_all", 0, T),
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

    write_extended_csv(ROOT / f"batch_{batch}.csv", rows)
    H.update_leaderboard(standard_rows(rows))

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

    # Walk-forward of batch 7 top 20, folds stop at hold_lo
    top20 = both[:20]
    print(f"walk-forward batch-7 top 20 both-window on pre-holdout (end_bar={hold_lo})", flush=True)
    folds, summaries = WF.walk_forward_shortlist(D, top20, original_split=split, end_bar=hold_lo)
    ranked = sorted(summaries, key=WF.robustness_key, reverse=True)
    WF.write_fold_csv(ROOT / "walkforward_batch7_folds.csv", folds)
    slim = [{k: v for k, v in r.items() if k != "_summary"} for r in ranked]
    WF.write_summary_csv(ROOT / "walkforward_batch7.csv", slim)

    # Holdout eval: robust top 10 + batch 7 top 20
    print("holdout evaluation of robust top 10 + batch-7 top 20", flush=True)
    caches = {}
    hold_rows = []
    robust = H.load_csv(ROOT / "robust_leaderboard.csv")[:10]
    for r in robust:
        r["_source"] = "robust_top10"
        is_m, purged, hold, pre, _ = eval_row_windows(r, D, caches, split, hold_lo, T)
        rec = holdout_record(r, hold, purged, is_m, pre, hold_lo, T, times, "includes_holdout (selected in batches 1-6)")
        rec["id"] = r["id"]
        hold_rows.append(rec)
        print(
            f"  {r['id']} hold_ret={hold['ret']} hold_profit={hold['profit']} "
            f"tr={hold['trades']} purged_oos_ret={purged['ret']}",
            flush=True,
        )
    # map holdout onto batch 7 top 20 from already-simulated rows
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
            "hold_start": _chi(times[hold_lo]),
            "hold_end": _chi(times[T - 1]),
            "hold_bars": int(T - hold_lo),
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

    write_hold_csv(ROOT / "holdout_eval.csv", hold_rows)

    # Update robust board: add batch-7 names that pass pre-holdout WF AND holdout profit > 0
    old = H.load_csv(ROOT / "robust_leaderboard.csv") if (ROOT / "robust_leaderboard.csv").exists() else []
    new_pass = []
    hold_by = {r["id"]: r for r in hold_rows if str(r["id"]).startswith("H7-")}
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

    top10_hold = [r for r in hold_rows if r["selected_on"].startswith("includes_holdout")]
    b7_hold = [r for r in hold_rows if r["selected_on"] == "pre_holdout_only"]
    n10_pos = sum(1 for r in top10_hold if r["hold_profitable"] == "true")
    n7_pos = sum(1 for r in b7_hold if r["hold_profitable"] == "true")
    dn10 = [r for r in top10_hold if "closing low" in r["entry"] or r["family"] == "washout_gate"]

    payload = {
        "holdout": {
            "bars": int(T - hold_lo),
            "start": _chi(times[hold_lo]),
            "end": _chi(times[T - 1]),
            "new_bars_since_batch6": int(max(0, T - 4297)),
            "note": (
                "No new bars arrived after batch 6. Holdout is the last 14 days "
                "already present in prior batches' OOS. Batch 7 was ranked only "
                "on data before this window."
            ),
        },
        "robust_top10_holdout": {
            "n": len(top10_hold),
            "n_profitable": n10_pos,
            "double_n_n": len(dn10),
            "double_n_profitable": sum(1 for r in dn10 if r["hold_profitable"] == "true"),
            "rows": top10_hold,
        },
        "batch7_top20_holdout": {
            "n": len(b7_hold),
            "n_profitable": n7_pos,
            "rows": b7_hold,
        },
        "n_robust_board": len(passed),
        "n_batch7_added_to_robust": len(new_pass),
    }
    with open(ROOT / "holdout_eval.json", "w") as f:
        json.dump(payload, f, indent=2)

    summary = {
        "batch": batch,
        "n": len(rows),
        "seconds": round(time.time() - t0, 1),
        "data": {
            "hours": int(T),
            "tokens": int(N),
            "split_bar": int(split),
            "holdout_lo": int(hold_lo),
            "holdout_start": _chi(times[hold_lo]),
            "holdout_end": _chi(times[T - 1]),
            "ranking": "IS + purged OOS (holdout excluded)",
        },
        "oos_profitable": len(oos_pos),
        "both_windows_profitable": len(both),
        "both_windows_solid": len(both_solid),
        "oos_only_flukes": len(flukes),
        "holdout_profitable": len(hold_pos),
        "both_and_holdout": len(both_and_hold),
        "top_both_windows": pack(both, 12, 0, holdout=True),
        "top_both_windows_solid": pack(both_solid, 12, 8, holdout=True),
        "top_both_and_holdout": pack(both_and_hold, 12, 0, holdout=True),
        "top_flukes": pack(flukes, 8, 0),
        "walkforward_pre_holdout_top20": [
            {
                "id": r["id"],
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
        f"robust top10 holdout profitable {n10_pos}/10, "
        f"{time.time()-t0:.1f}s",
        flush=True,
    )
    return summary


if __name__ == "__main__":
    run()
