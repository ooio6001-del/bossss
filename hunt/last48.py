#!/usr/bin/env python3
"""Replay hunt + moonshot configs over the last 1h, 24h, and 48h only.

Does not modify /workspace/solstrats. Refreshed candles are written under
data_last48/. Positions start flat at each window start (pre-window signals
are cleared so one-shot moonshots can still fire inside the window).
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path("/workspace/solstrats_hunt")
MAIN = Path("/workspace/solstrats")
SRC_DATA = MAIN / "data"
DST_DATA = ROOT / "data_last48"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MAIN))

from solstrats import costs as K  # noqa: E402
from solstrats import engine as E  # noqa: E402
from solstrats import grid as G  # noqa: E402
from solstrats.describe import entry_text as moon_entry_text  # noqa: E402
from solstrats.describe import exit_text, sizing_text  # noqa: E402
from solstrats.http import Client, HttpError  # noqa: E402
from run_backtest import prep  # noqa: E402
import wf_lib  # noqa: E402

CHI = ZoneInfo("America/Chicago")
GT = "https://api.geckoterminal.com/api/v2/networks/solana"
NONE = E.NONE
REFRESH_BUDGET_S = 270  # stay inside the ~5 minute refresh budget
STALE_MAX_H = 7 * 24  # do not stitch huge gaps onto dead launch series
WINDOWS_H = (1, 24, 48)
MOON_BALS = (10.0, 100.0)


def chi(ts: float) -> str:
    return datetime.fromtimestamp(float(ts), tz=CHI).strftime("%Y-%m-%d %I:%M %p %Z")


def now_chi() -> str:
    return datetime.now(CHI).strftime("%Y-%m-%d %I:%M %p %Z")


def _atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f)
    tmp.replace(path)


def _should_refresh(meta: dict, rows: list, latest_complete: int, now_ts: float) -> bool:
    group = meta.get("group")
    if not rows:
        return group in ("established", "benchmark")
    end = int(rows[-1][0])
    if end >= latest_complete:
        return False
    age_h = (now_ts - end) / 3600.0
    if group in ("established", "benchmark"):
        return True
    return age_h <= STALE_MAX_H


def refresh_ohlcv(deadline: float) -> dict:
    """Pull recent hourly candles into DST_DATA without touching SRC_DATA."""
    dst_ohlcv = DST_DATA / "ohlcv"
    reuse = dst_ohlcv.exists() and any(dst_ohlcv.glob("*.json"))
    if not reuse:
        if DST_DATA.exists():
            shutil.rmtree(DST_DATA)
        dst_ohlcv.mkdir(parents=True)
        src_onchain = SRC_DATA / "onchain.json"
        if src_onchain.exists() and not (DST_DATA / "onchain.json").exists():
            os.symlink(src_onchain, DST_DATA / "onchain.json")
    else:
        src_onchain = SRC_DATA / "onchain.json"
        if src_onchain.exists() and not (DST_DATA / "onchain.json").exists():
            os.symlink(src_onchain, DST_DATA / "onchain.json")
        print("refresh: reusing existing data_last48, only filling still-stale pools", flush=True)

    src_files = sorted((SRC_DATA / "ohlcv").glob("*.json"))
    now_ts = time.time()
    latest_complete = int(now_ts // 3600) * 3600
    # GeckoTerminal's current hour bar may be incomplete; still useful, but
    # the simulation window ends at the last complete hour after the merge.

    targets = []
    for fn in src_files:
        try:
            with open(fn) as f:
                d = json.load(f)
        except (OSError, ValueError):
            continue
        meta = d.get("meta") or {}
        rows = d.get("ohlcv") or []
        dst = DST_DATA / "ohlcv" / fn.name
        if dst.exists() or dst.is_symlink():
            # Already staged from a prior run. Refresh again only if still stale
            # and the staged file is just a symlink back to the original cache.
            if not dst.is_symlink():
                continue
            try:
                with open(dst) as f:
                    staged = json.load(f)
                if not _should_refresh(staged.get("meta") or meta, staged.get("ohlcv") or [], latest_complete, now_ts):
                    continue
            except (OSError, ValueError):
                pass
            targets.append((fn, dst, meta, rows))
            continue
        if _should_refresh(meta, rows, latest_complete, now_ts):
            targets.append((fn, dst, meta, rows))
        else:
            os.symlink(fn, dst)

    stats = {
        "attempted": 0,
        "updated": 0,
        "unchanged": 0,
        "failed": 0,
        "skipped_stale": len(src_files) - len(targets),
        "failures": [],
        "requests": 0,
        "retries": 0,
        "http_429": 0,
        "timed_out": False,
        "latest_complete_hour": chi(latest_complete),
    }
    client = Client(cache_dir=None, timeout=20.0)
    print(
        f"refresh: {len(targets)} pools to update, {stats['skipped_stale']} already current or too stale; "
        f"latest complete hour {chi(latest_complete)}",
        flush=True,
    )
    for fn, dst, meta, rows in targets:
        if time.time() >= deadline:
            stats["timed_out"] = True
            if not dst.exists() and not dst.is_symlink():
                os.symlink(fn, dst)
            continue
        pool = meta.get("pool")
        sym = meta.get("sym") or "?"
        stats["attempted"] += 1
        try:
            raw = client.get(
                "geckoterminal",
                f"{GT}/pools/{pool}/ohlcv/hour",
                params={"aggregate": 1, "limit": 1000, "currency": "usd", "token": "base"},
            )
            fresh = raw["data"]["attributes"]["ohlcv_list"]
        except (HttpError, KeyError, TypeError) as e:
            stats["failed"] += 1
            stats["failures"].append({"sym": sym, "pool": pool, "error": type(e).__name__})
            if not dst.exists() and not dst.is_symlink():
                os.symlink(fn, dst)
            print(f"  fail {sym}: {type(e).__name__}", flush=True)
            continue
        merged = {int(r[0]): r for r in rows}
        added = 0
        for r in fresh:
            ts = int(r[0])
            if ts not in merged:
                added += 1
            merged[ts] = r
        new_rows = [merged[k] for k in sorted(merged)]
        if added or (new_rows and rows and int(new_rows[-1][0]) != int(rows[-1][0])):
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            _atomic_write_json(dst, {"meta": meta, "ohlcv": new_rows})
            stats["updated"] += 1
            print(
                f"  {sym}: +{added} bars, end {chi(new_rows[-1][0])} n={len(new_rows)}",
                flush=True,
            )
        else:
            os.symlink(fn, dst)
            stats["unchanged"] += 1
    stats["requests"] = int(client.stats.get("requests", 0))
    stats["retries"] = int(client.stats.get("retries", 0))
    stats["http_429"] = int(client.stats.get("429", 0))
    print(
        f"refresh done: updated={stats['updated']} unchanged={stats['unchanged']} "
        f"failed={stats['failed']} timeout={stats['timed_out']} "
        f"http requests={stats['requests']} 429s={stats['http_429']}",
        flush=True,
    )
    return stats


def coverage(D: dict) -> dict:
    times = D["times"]
    C = D["C"]
    T, N = C.shape
    last = D["last"]
    t_end = int(times[-1])
    n_at_end = int((last == T - 1).sum())
    n_valid = int((last >= 0).sum())
    return {
        "n_tokens": int(N),
        "n_with_data": n_valid,
        "n_at_end_bar": n_at_end,
        "t0": float(times[0]),
        "t_end": float(t_end),
        "t0_chi": chi(times[0]),
        "t_end_chi": chi(t_end),
        "hours": int(T),
    }


def window_bounds(D: dict, hours: int) -> tuple[int, int, float, float]:
    """Return (t_lo, t_hi, start_ts, end_ts) for the last `hours` complete bars.

    t_hi is exclusive and equals T (the last complete hour is times[T-1]).
    The window contains exactly `hours` fill bars when T >= hours.
    """
    T = int(D["T"])
    times = D["times"]
    n = min(hours, T)
    t_lo = T - n
    t_hi = T
    return t_lo, t_hi, float(times[t_lo]), float(times[t_hi - 1])


def mask_pre_window(sig: np.ndarray, t_lo: int) -> np.ndarray:
    """Keep only signals whose next-bar fill is inside [t_lo, ...). Starts flat."""
    s = np.array(sig, copy=True, dtype=np.bool_)
    if t_lo > 1:
        s[: t_lo - 1] = False
    return s


def pack_metrics(m: np.ndarray, bal: float) -> dict:
    ntr = int(m[4])
    wins = int(m[5])
    return {
        "end_bal": round(float(m[0]), 4),
        "profit": round(float(m[0] - bal), 4),
        "ret": round(float(m[1]), 6),
        "ret_pct": round(float(m[1]) * 100.0, 4),
        "trades": ntr,
        "wins": wins,
        "win_rate": round(float(m[13]), 4),
        "win_rate_pct": round(float(m[13]) * 100.0, 2) if ntr else 0.0,
        "max_dd": round(float(m[2]), 4),
        "fees": round(float(m[11]), 4),
        "skipped": int(m[10]),
        "best_trade": round(float(m[8]), 4) if ntr else 0.0,
        "worst_trade": round(float(m[9]), 4) if ntr else 0.0,
    }


def simulate_trades(D, tok, ti, to, pin, pout, t_lo, t_hi, frac, maxpos, bal):
    vol_entry = D["V"][np.maximum(ti - 1, 0), tok] if tok.size else np.zeros(0)
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
        t_lo,
        t_hi,
        float(bal),
        float(frac),
        int(maxpos),
        K.PCT_COST,
        K.FIXED_SOL,
        K.RENT_SOL,
        K.RENT_UNRECOVERED,
        K.IMPACT_K,
        K.FILL_CAP,
        empty,
    )


def gen_window_trades(sig, e, D, t_lo, one_shot):
    s = mask_pre_window(sig, t_lo)
    return E.gen_trades(
        s,
        np.asarray(e, dtype=np.float64),
        D["O"],
        D["H"],
        D["L"],
        D["C"],
        D["last"],
        D["supply"],
        bool(one_shot),
    )


def result_row(base: dict, metrics: dict, window_h: int) -> dict:
    out = dict(base)
    out.update(metrics)
    out["window_h"] = window_h
    return out


def load_hunt_rows():
    with open(ROOT / "leaderboard.csv") as f:
        board = list(csv.DictReader(f))
    with open(ROOT / "robust_leaderboard.csv") as f:
        robust = list(csv.DictReader(f))
    robust_ids = {r["id"] for r in robust}
    top = board[:2000]
    by_key = {}
    for r in top:
        by_key[(r["id"], r["balance"], r["sizing"])] = r
    extra = []
    for r in robust:
        k = (r["id"], r["balance"], r["sizing"])
        if k not in by_key:
            by_key[k] = r
            extra.append(r)
    rows = list(by_key.values())
    return rows, robust, robust_ids, extra


def eval_hunt(D, rows, windows, caches):
    """Evaluate each hunt row on every window. Cache signals by (batch, family, params)."""
    sig_cache = {}
    results = {h: [] for h, _lo, _hi in windows}
    errors = []
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        key = (str(row.get("batch")), row["family"], row["params"])
        if key not in sig_cache:
            try:
                sig_cache[key] = wf_lib.signal_from_row(row, D, caches)
            except Exception as e:  # noqa: BLE001 - one bad row must not kill the run
                errors.append({"id": row.get("id"), "family": row.get("family"), "error": f"{type(e).__name__}: {e}"})
                sig_cache[key] = None
        sig = sig_cache[key]
        if sig is None:
            continue
        try:
            e = json.loads(row["exit_params"]) if isinstance(row["exit_params"], str) else row["exit_params"]
            frac, maxpos = wf_lib.parse_sizing(row["sizing"])
            bal = float(row["balance"])
        except (ValueError, json.JSONDecodeError, KeyError) as e:
            errors.append({"id": row.get("id"), "family": row.get("family"), "error": f"{type(e).__name__}: {e}"})
            continue
        base = {
            "source": "hunt",
            "id": row["id"],
            "batch": row.get("batch"),
            "family": row["family"],
            "entry": row.get("entry"),
            "exit": row.get("exit"),
            "sizing": row.get("sizing"),
            "balance": bal,
            "robust": False,
        }
        # Generate once from the earliest (longest) window start; shorter windows
        # just restrict entry bars so they still start flat.
        t_lo_min = min(w[1] for w in windows)
        tok, ti, to, pin, pout, _rs = gen_window_trades(sig, e, D, t_lo_min, False)
        for hours, t_lo, t_hi in windows:
            m = simulate_trades(D, tok, ti, to, pin, pout, t_lo, t_hi, frac, maxpos, bal)
            rec = result_row(base, pack_metrics(m, bal), hours)
            results[hours].append(rec)
        if i % 200 == 0 or i == len(rows):
            print(f"  hunt {i}/{len(rows)}  {time.time() - t0:.1f}s  signal-cache {len(sig_cache)}  errors {len(errors)}", flush=True)
    return results, errors


def moonshot_combos():
    out = []
    for fam, p, es, one in G.entries():
        if fam != "moonshot":
            continue
        for e in G.EXIT_SETS[es]:
            out.append((tuple(p), tuple(e), es, one))
    return out


def eval_moonshots(D, windows):
    combos = moonshot_combos()
    # unique entry params
    by_p = {}
    for p, e, es, one in combos:
        by_p.setdefault(p, []).append((e, es, one))
    print(f"  moonshot entries {len(by_p)}  entry-exit combos {len(combos)}  balances {MOON_BALS}  sizing {len(G.SIZING)}", flush=True)
    results = {h: [] for h, _lo, _hi in windows}
    t0 = time.time()
    n_sig = 0
    mid = 0
    for p, exits in by_p.items():
        p_arr = np.asarray(p, dtype=np.float64)
        hot = D["hot"][24]
        sig = E.signals(
            E.F["moonshot"],
            p_arr,
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
        n_sig += 1
        entry = moon_entry_text("moonshot", p)
        for e, es, one in exits:
            e_arr = np.asarray(e, dtype=np.float64)
            tp = float(e[0])
            tp_pct = None if tp >= NONE / 2 else round(tp * 100.0, 4)
            base_common = {
                "source": "moonshot",
                "id": f"MS-{n_sig:03d}-{es}-{int(tp_pct or 0)}",
                "batch": "sweep",
                "family": "moonshot",
                "entry": entry,
                "exit": exit_text(e),
                "params": list(p),
                "exit_params": list(e),
                "exit_set": es,
                "tp_pct": tp_pct,
            }
            t_lo_min = min(w[1] for w in windows)
            tok, ti, to, pin, pout, _rs = gen_window_trades(sig, e_arr, D, t_lo_min, True)
            vol_entry = D["V"][np.maximum(ti - 1, 0), tok] if tok.size else np.zeros(0)
            ncfg = len(G.SIZING) * len(MOON_BALS)
            offs = np.array([0, tok.size], dtype=np.int64)
            cfg_combo = np.zeros(ncfg, dtype=np.int64)
            cfg_bal = np.array([b for _s in G.SIZING for b in MOON_BALS], dtype=np.float64)
            cfg_frac = np.array([s[0] for s in G.SIZING for _b in MOON_BALS], dtype=np.float64)
            cfg_maxpos = np.array([s[1] for s in G.SIZING for _b in MOON_BALS], dtype=np.int64)
            for hours, t_lo, t_hi in windows:
                res = E.simulate_many(
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
                    t_lo,
                    t_hi,
                    K.PCT_COST,
                    K.FIXED_SOL,
                    K.RENT_SOL,
                    K.RENT_UNRECOVERED,
                    K.IMPACT_K,
                    K.FILL_CAP,
                )
                for i in range(ncfg):
                    frac, maxpos = cfg_frac[i], int(cfg_maxpos[i])
                    bal = float(cfg_bal[i])
                    base = dict(base_common)
                    base["id"] = f"MS-{n_sig:03d}-{es}-tp{int(tp_pct or 0)}-f{int(frac*100)}-n{maxpos}-b{int(bal)}"
                    base["sizing"] = sizing_text(frac, maxpos)
                    base["balance"] = bal
                    rec = result_row(base, pack_metrics(res[i], bal), hours)
                    results[hours].append(rec)
                    mid += 1
        if n_sig % 6 == 0 or n_sig == len(by_p):
            print(f"  moonshot signals {n_sig}/{len(by_p)}  rows {mid}  {time.time() - t0:.1f}s", flush=True)
    return results


def tag_robust(rows, robust_ids):
    for r in rows:
        r["robust"] = r.get("source") == "hunt" and r.get("id") in robust_ids
    return rows


def top_n(rows, key, n, require_trades=False):
    pool = [r for r in rows if (not require_trades or int(r["trades"]) > 0)]
    return sorted(pool, key=lambda r: (float(r[key]), float(r["ret"]), int(r["trades"])), reverse=True)[:n]


def slim(r, extra=()):
    keys = [
        "id",
        "source",
        "family",
        "entry",
        "exit",
        "sizing",
        "balance",
        "profit",
        "ret_pct",
        "end_bal",
        "trades",
        "wins",
        "win_rate_pct",
        "max_dd",
        "fees",
        "skipped",
        "robust",
        "tp_pct",
        "window_h",
    ]
    keys.extend(k for k in extra if k not in keys)
    return {k: r.get(k) for k in keys if k in r or k in extra}


def moon_tp_summary(rows):
    """Best take-profit level among moonshot configs in this window."""
    by_tp = {}
    for r in rows:
        if r.get("source") != "moonshot":
            continue
        tp = r.get("tp_pct")
        if tp is None:
            bucket = "uncapped"
        else:
            bucket = f"{int(tp) if float(tp).is_integer() else tp}%"
        s = by_tp.setdefault(
            bucket,
            {
                "tp": bucket,
                "tp_pct": tp,
                "n": 0,
                "with_trades": 0,
                "profitable": 0,
                "profits": [],
                "rets": [],
                "best": None,
            },
        )
        s["n"] += 1
        if int(r["trades"]) > 0:
            s["with_trades"] += 1
            s["profits"].append(float(r["profit"]))
            s["rets"].append(float(r["ret"]))
            if s["best"] is None or float(r["profit"]) > float(s["best"]["profit"]):
                s["best"] = slim(r)
        if float(r["profit"]) > 0:
            s["profitable"] += 1
    out = []
    for s in by_tp.values():
        rec = {
            "tp": s["tp"],
            "tp_pct": s["tp_pct"],
            "n_configs": s["n"],
            "with_trades": s["with_trades"],
            "profitable": s["profitable"],
            "median_profit_traded": round(float(np.median(s["profits"])), 4) if s["profits"] else None,
            "median_ret_pct_traded": round(float(np.median(s["rets"])) * 100.0, 4) if s["rets"] else None,
            "best": s["best"],
        }
        out.append(rec)
    # rank: prefer best single profit, then median among those that traded
    out.sort(
        key=lambda x: (
            -float(x["best"]["profit"]) if x["best"] else 1e18,
            -(x["median_profit_traded"] if x["median_profit_traded"] is not None else -1e18),
            -x["with_trades"],
        )
    )
    return out


def summarize_window(hours, rows, robust_ids, start_ts, end_ts, t_lo, t_hi):
    hunt = [r for r in rows if r["source"] == "hunt"]
    moon = [r for r in rows if r["source"] == "moonshot"]
    robust = [r for r in hunt if r.get("id") in robust_ids]
    traded = [r for r in rows if int(r["trades"]) > 0]
    tp = moon_tp_summary(rows)
    best_robust = top_n(robust, "profit", 1)
    return {
        "hours": hours,
        "bar_start": chi(start_ts),
        "bar_end": chi(end_ts),
        "start_ts": start_ts,
        "end_ts": end_ts,
        "t_lo": t_lo,
        "t_hi": t_hi,
        "n_bars": t_hi - t_lo,
        "n_configs": len(rows),
        "n_hunt": len(hunt),
        "n_moonshot": len(moon),
        "n_robust": len(robust),
        "n_with_trades": len(traded),
        "n_profitable": sum(1 for r in rows if float(r["profit"]) > 0),
        "top3_by_profit": [slim(r) for r in top_n(rows, "profit", 3)],
        "top3_by_return": [slim(r) for r in top_n(rows, "ret", 3)],
        "top5_by_profit": [slim(r) for r in top_n(rows, "profit", 5)],
        "top5_by_return": [slim(r) for r in top_n(rows, "ret", 5)],
        "best_robust": slim(best_robust[0]) if best_robust else None,
        "best_robust_by_return": slim(top_n(robust, "ret", 1)[0]) if robust else None,
        "robust_with_trades": sum(1 for r in robust if int(r["trades"]) > 0),
        "best_moonshot_tp": tp[0] if tp else None,
        "moonshot_tp_table": tp,
        "note": (
            "1-hour window is mostly a check on whether a position was opened in that hour; trade counts will be small or zero."
            if hours == 1
            else "48 hours (and even 24) is a tiny sample; do not treat a winner here as a validated edge."
        ),
    }


def write_csv(path: Path, rows):
    if not rows:
        return
    fields = [
        "window_h",
        "id",
        "source",
        "family",
        "robust",
        "balance",
        "profit",
        "ret_pct",
        "end_bal",
        "trades",
        "wins",
        "win_rate_pct",
        "max_dd",
        "fees",
        "skipped",
        "tp_pct",
        "sizing",
        "entry",
        "exit",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def warmup(D):
    empty = np.zeros((D["T"], D["N"]), dtype=np.bool_)
    E.gen_trades(
        empty,
        np.asarray((1.0, 0.35, NONE, 0.0, 240.0, NONE, 1.0), dtype=np.float64),
        D["O"],
        D["H"],
        D["L"],
        D["C"],
        D["last"],
        D["supply"],
        False,
    )
    tok = np.zeros(0, dtype=np.int64)
    z = np.zeros(0)
    simulate_trades(D, tok, tok, tok, z, z, 0, D["T"], 0.1, 1, 10.0)


def main():
    t_start = time.time()
    print(f"last48 start {now_chi()}", flush=True)
    dst_ready = (DST_DATA / "ohlcv").exists() and any((DST_DATA / "ohlcv").glob("*.json"))
    if dst_ready:
        n_real = sum(1 for q in (DST_DATA / "ohlcv").glob("*.json") if not q.is_symlink())
        n_link = sum(1 for q in (DST_DATA / "ohlcv").glob("*.json") if q.is_symlink())
        print(
            f"reusing existing data_last48 ({n_real} newly fetched files, {n_link} cached symlinks); "
            "skipping another GeckoTerminal sweep after the first pass hit 429s and timed out",
            flush=True,
        )
        refresh_stats = {
            "reused": True,
            "attempted": 0,
            "updated": n_real,
            "unchanged": 0,
            "failed": 0,
            "skipped_stale": n_link,
            "failures": [],
            "requests": 0,
            "retries": 0,
            "http_429": 0,
            "timed_out": True,
            "note": (
                "First refresh pass updated 8 pools (including SOL) to 2026-11-13 03:00 PM CST, "
                "then exhausted the ~5 minute budget after GeckoTerminal 429s "
                "(92 requests, 65 rate-limit responses). This run reuses that snapshot "
                "and does not modify /workspace/solstrats/data."
            ),
        }
    else:
        refresh_stats = refresh_ohlcv(time.time() + REFRESH_BUDGET_S)

    print("loading data", flush=True)
    D = prep(str(DST_DATA))
    cov = coverage(D)
    print(
        f"data {cov['hours']} hours x {cov['n_tokens']} tokens; "
        f"{cov['n_at_end_bar']} have the end bar; "
        f"{cov['t0_chi']} -> {cov['t_end_chi']}",
        flush=True,
    )
    warmup(D)

    windows_meta = []
    windows = []
    for h in WINDOWS_H:
        t_lo, t_hi, ts0, ts1 = window_bounds(D, h)
        windows.append((h, t_lo, t_hi))
        windows_meta.append(
            {"hours": h, "t_lo": t_lo, "t_hi": t_hi, "start": chi(ts0), "end": chi(ts1), "n_bars": t_hi - t_lo}
        )
        print(f"window {h}h: {chi(ts0)} -> {chi(ts1)}  bars [{t_lo}, {t_hi})  n={t_hi - t_lo}", flush=True)

    hunt_rows, robust, robust_ids, extra = load_hunt_rows()
    print(
        f"hunt rows {len(hunt_rows)} (top 2000 + {len(extra)} robust not already in top 2000); "
        f"robust names {len(robust_ids)}",
        flush=True,
    )
    caches = {}
    print("evaluating hunt rows", flush=True)
    hunt_res, hunt_errors = eval_hunt(D, hunt_rows, windows, caches)
    print("evaluating moonshot grid", flush=True)
    moon_res = eval_moonshots(D, windows)

    summaries = []
    all_csv = []
    for (h, t_lo, t_hi), meta in zip(windows, windows_meta):
        rows = hunt_res[h] + moon_res[h]
        tag_robust(rows, robust_ids)
        # keep a compact leaderboard (traded or top profits) for the csv
        traded = [r for r in rows if int(r["trades"]) > 0]
        top_p = top_n(rows, "profit", 100)
        top_r = top_n(rows, "ret", 100)
        seen = set()
        for r in traded + top_p + top_r:
            k = (r["window_h"], r["id"], r["balance"], r.get("sizing"))
            if k in seen:
                continue
            seen.add(k)
            all_csv.append(r)
        summ = summarize_window(h, rows, robust_ids, meta["start"] and windows_meta[WINDOWS_H.index(h)] and D["times"][t_lo], D["times"][t_hi - 1], t_lo, t_hi)
        # fix start/end using numeric times
        summ["bar_start"] = chi(D["times"][t_lo])
        summ["bar_end"] = chi(D["times"][t_hi - 1])
        summ["start_ts"] = float(D["times"][t_lo])
        summ["end_ts"] = float(D["times"][t_hi - 1])
        summaries.append(summ)
        print(
            f"window {h}h: {summ['n_with_trades']} of {summ['n_configs']} configs traded; "
            f"profitable {summ['n_profitable']}",
            flush=True,
        )

    out = {
        "generated_at": now_chi(),
        "generated_ts": time.time(),
        "elapsed_s": round(time.time() - t_start, 1),
        "timezone": "America/Chicago",
        "cost_model": "solstrats.costs (same PCT_COST / FIXED_SOL / RENT / IMPACT_K / FILL_CAP)",
        "positions_start_flat": True,
        "refresh": refresh_stats,
        "coverage": cov,
        "windows": summaries,
        "hunt_errors": hunt_errors,
        "notes": [
            "Windows use the last N complete hourly bars ending at coverage.t_end. The current incomplete hour is not simulated.",
            "Positions start flat at each window start; pre-window signals are cleared so inherited positions are not carried in.",
            "48h/24h/1h samples are tiny. A large profit here is not evidence the strategy works out of sample.",
            "The 1-hour window mostly asks whether a position was opened in that hour; expect very few trades.",
            "Moonshot take-profit is e[0]*100%. Uncapped (runner) exits have no fixed TP.",
            "Existing files under /workspace/solstrats were not modified; refreshed candles live in data_last48/.",
        ],
    }
    out_json = ROOT / "last48_results.json"
    out_csv = ROOT / "last48_top.csv"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    write_csv(out_csv, all_csv)
    print(f"wrote {out_json} and {out_csv} in {out['elapsed_s']}s", flush=True)

    # compact stdout report for the parent
    print("\n========== LAST-N-HOURS REPORT ==========", flush=True)
    print(f"generated {out['generated_at']}", flush=True)
    print(f"data end {cov['t_end_chi']}  tokens at end bar {cov['n_at_end_bar']}/{cov['n_tokens']}", flush=True)
    print(
        f"refresh updated={refresh_stats['updated']} failed={refresh_stats['failed']} "
        f"timeout={refresh_stats['timed_out']} 429s={refresh_stats['http_429']}",
        flush=True,
    )
    for s in summaries:
        print(f"\n--- {s['hours']}h  {s['bar_start']} -> {s['bar_end']}  ({s['n_bars']} bars) ---", flush=True)
        print(f"configs {s['n_configs']}  with trades {s['n_with_trades']}  profitable {s['n_profitable']}", flush=True)
        print("top 3 by $ profit:", flush=True)
        for i, r in enumerate(s["top3_by_profit"], 1):
            print(
                f"  {i}. {r['id']}  {r['family']}  bal ${r['balance']:g}  "
                f"P&L ${r['profit']:+.4f} ({r['ret_pct']:+.2f}%)  "
                f"trades {r['trades']} win {r['win_rate_pct']:.0f}%  {r['sizing']}",
                flush=True,
            )
            print(f"     entry: {r['entry']}", flush=True)
            print(f"     exit:  {r['exit']}", flush=True)
        print("top 3 by % return:", flush=True)
        for i, r in enumerate(s["top3_by_return"], 1):
            print(
                f"  {i}. {r['id']}  {r['family']}  bal ${r['balance']:g}  "
                f"P&L ${r['profit']:+.4f} ({r['ret_pct']:+.2f}%)  "
                f"trades {r['trades']} win {r['win_rate_pct']:.0f}%  {r['sizing']}",
                flush=True,
            )
        if s["hours"] == 48:
            print("top 5 by $ profit (48h):", flush=True)
            for i, r in enumerate(s["top5_by_profit"], 1):
                print(
                    f"  {i}. {r['id']}  {r['family']}  bal ${r['balance']:g}  "
                    f"P&L ${r['profit']:+.4f} ({r['ret_pct']:+.2f}%)  trades {r['trades']} win {r['win_rate_pct']:.0f}%",
                    flush=True,
                )
            print("top 5 by % return (48h):", flush=True)
            for i, r in enumerate(s["top5_by_return"], 1):
                print(
                    f"  {i}. {r['id']}  {r['family']}  bal ${r['balance']:g}  "
                    f"P&L ${r['profit']:+.4f} ({r['ret_pct']:+.2f}%)  trades {r['trades']} win {r['win_rate_pct']:.0f}%",
                    flush=True,
                )
        br = s["best_robust"]
        if br:
            print(
                f"best robust: {br['id']}  {br['family']}  bal ${br['balance']:g}  "
                f"P&L ${br['profit']:+.4f} ({br['ret_pct']:+.2f}%)  trades {br['trades']} win {br['win_rate_pct']:.0f}%",
                flush=True,
            )
            print(f"  entry: {br['entry']}", flush=True)
            print(f"  exit:  {br['exit']}", flush=True)
        else:
            print("best robust: none", flush=True)
        tp = s["best_moonshot_tp"]
        if tp and tp.get("best"):
            print(
                f"best moonshot TP: {tp['tp']}  best P&L ${tp['best']['profit']:+.4f} "
                f"({tp['best']['ret_pct']:+.2f}%)  trades {tp['best']['trades']}  "
                f"level with_trades={tp['with_trades']} profitable={tp['profitable']} "
                f"median_profit_traded={tp['median_profit_traded']}",
                flush=True,
            )
        elif tp:
            print(f"best moonshot TP: {tp['tp']}  no trades at this level", flush=True)
        else:
            print("best moonshot TP: no moonshot configs", flush=True)
        print(s["note"], flush=True)
    if hunt_errors:
        print(f"hunt signal errors: {len(hunt_errors)} (see last48_results.json)", flush=True)
    return out


if __name__ == "__main__":
    main()
