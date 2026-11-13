#!/usr/bin/env python3
"""SOLSTRATS LIVE PAPER trader (no wallet, no real orders, virtual $ only).

Each cycle (hourly, a few minutes after the hour):
  1. refresh hourly candles from GeckoTerminal for every pool still trading (token-bucket rate limit, backoff on 429),
     dropping the still-forming current hour so only COMPLETE bars are used (same as the backtest).
  2. rebuild each strategy's entry signal with the exact hunter code (wf_lib.signal_from_row), clear every signal before
     the live start, generate trades with engine.gen_trades (same stop/trail/time exits), and run the portfolio with the
     same sizing + cost model as engine.simulate (DEX fee+slippage per venue, price impact, priority/base fee, token rent,
     5%-of-bar-volume fill cap). Open positions are marked at the last complete close, exit costs included.
  3. write state.json, trades.csv, equity.csv, events.log.
Restart-safe: the live start time and config live in state.json; every cycle is a deterministic replay from that start,
so a crash/restart cannot double-count or lose trades.
"""
from __future__ import annotations
import csv, json, math, os, shutil, sys, time, traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path("/workspace/solstrats_live")
DATA = ROOT / "data"
sys.path.insert(0, "/workspace/solstrats_hunt"); sys.path.insert(0, "/workspace/solstrats")
sys.argv = [sys.argv[0]] + sys.argv[1:]
import numpy as np
from solstrats import costs as K, engine as E
from solstrats.http import Client, HttpError
from run_backtest import prep
import wf_lib

CHI = ZoneInfo("America/Chicago")
GT = "https://api.geckoterminal.com/api/v2/networks/solana"
STALE_MAX_H = 7 * 24
REASON = {E.R_SL: "stop", E.R_TP: "take-profit", E.R_TRAIL: "trailing stop", E.R_TIME: "time exit",
          E.R_MCT: "mcap target", E.R_CRASH: "crash exit", E.R_END: "data ended"}

def chi(ts): return datetime.fromtimestamp(float(ts), CHI).strftime("%Y-%m-%d %I:%M %p %Z")
def log(msg):
    line = f"[{datetime.now(CHI).strftime('%Y-%m-%d %I:%M:%S %p %Z')}] {msg}"
    print(line, flush=True)
    with open(ROOT / "events.log", "a") as f: f.write(line + "\n")
def atomic_json(p, obj):
    tmp = Path(str(p) + ".tmp"); tmp.write_text(json.dumps(obj, indent=1, default=float)); tmp.replace(p)

def init_data():
    if (DATA / "ohlcv").exists(): return
    src = Path("/workspace/solstrats_hunt/data_last48")
    (DATA / "ohlcv").mkdir(parents=True)
    for fn in (src / "ohlcv").glob("*.json"):
        shutil.copyfile(fn.resolve(), DATA / "ohlcv" / fn.name)
    shutil.copyfile((src / "onchain.json").resolve(), DATA / "onchain.json")
    log(f"data dir initialised from {src}")

def refresh(budget_s=1800):
    now = time.time(); cur_hour = int(now // 3600) * 3600   # bars with ts >= cur_hour are still forming
    client = Client(cache_dir=None, timeout=20.0)
    st = {"updated": 0, "unchanged": 0, "failed": 0, "skipped_dead": 0, "timed_out": False}
    deadline = now + budget_s
    for fn in sorted((DATA / "ohlcv").glob("*.json")):
        d = json.loads(fn.read_text()); meta = d["meta"]; rows = d.get("ohlcv") or []
        rows = [r for r in rows if int(r[0]) < cur_hour]
        end = int(rows[-1][0]) if rows else 0
        if end >= cur_hour - 3600: st["unchanged"] += 1; continue
        if meta.get("group") not in ("established", "benchmark") and (now - end) / 3600 > STALE_MAX_H:
            st["skipped_dead"] += 1; continue
        if time.time() > deadline: st["timed_out"] = True; continue
        try:
            raw = client.get("geckoterminal", f"{GT}/pools/{meta['pool']}/ohlcv/hour",
                             params={"aggregate": 1, "limit": 72, "currency": "usd", "token": "base"})
            fresh = raw["data"]["attributes"]["ohlcv_list"]
        except (HttpError, KeyError, TypeError, ValueError) as e:
            st["failed"] += 1; log(f"refresh fail {meta.get('sym')}: {type(e).__name__}"); continue
        merged = {int(r[0]): r for r in rows}
        for r in fresh:
            if int(r[0]) < cur_hour: merged[int(r[0])] = r
        new = [merged[k] for k in sorted(merged)]
        if new and new[-1][0] != (rows[-1][0] if rows else None):
            tmp = fn.with_suffix(".tmp"); tmp.write_text(json.dumps({"meta": meta, "ohlcv": new})); tmp.replace(fn)
            st["updated"] += 1
        else:
            st["unchanged"] += 1
    st["http"] = dict(client.stats)
    return st

def sim_log(D, tok, ti, to, pin, pout, rs, isopen, t_lo, t_hi, bal, frac, maxpos):
    """Line-by-line Python port of window.simulate_window that also returns the trade list."""
    order = np.argsort(ti * 1.0 + tok * 1e-7)
    cash = bal; slots = [None] * maxpos; trades = []; peak = bal; maxdd = 0.0; skipped = []
    times = D["times"]; metas = D["metas"]
    def close_before(t_now):
        nonlocal cash, peak, maxdd
        while True:
            live = [s for s in slots if s is not None and s["exit_bar"] < t_now]
            if not live: return
            s = min(live, key=lambda x: x["exit_bar"]); slots[slots.index(s)] = None
            cash += s["val"] + s["lock"] * (1 - K.RENT_UNRECOVERED)
            pnl = s["val"] - s["cost"] - s["lock"] * K.RENT_UNRECOVERED
            s["pnl"] = pnl; s["cash_after"] = cash
            trades.append(s)
            eq = cash + sum(o["cost"] + o["lock"] for o in slots if o is not None)
            peak = max(peak, eq); maxdd = max(maxdd, 1 - eq / peak if peak > 0 else 0)
    for q in order:
        t_now = int(ti[q])
        if t_now < t_lo or t_now >= t_hi: continue
        close_before(t_now)
        sym = metas[int(tok[q])].get("sym", "?")
        if sum(s is not None for s in slots) >= maxpos:
            skipped.append({"bar": t_now, "sym": sym, "why": "max positions open"}); continue
        eq = cash + sum(o["cost"] + o["lock"] for o in slots if o is not None)
        size = frac * eq; sp = D["solpx"][t_now]; fix = K.FIXED_SOL * sp; lock = K.RENT_SOL * sp
        if size + lock > cash: size = cash - lock
        if size <= fix * 2: skipped.append({"bar": t_now, "sym": sym, "why": "not enough cash"}); continue
        ve = float(D["V"][max(t_now - 1, 0), tok[q]])
        if size > K.FILL_CAP * ve: skipped.append({"bar": t_now, "sym": sym, "why": "size > 5% of bar volume"}); continue
        pc = K.PCT_COST[D["venue"][tok[q]]] + K.IMPACT_K * size / (ve + 1.0)
        gross = pout[q] / pin[q]
        val = (size - fix) * (1 - pc) * gross * (1 - pc) - fix
        cash -= size + lock
        rec = {"sym": sym, "mint": metas[int(tok[q])].get("mint"), "pool": metas[int(tok[q])].get("pool"),
               "entry_bar": t_now, "exit_bar": int(to[q]), "entry_time": chi(times[t_now]),
               "exit_time": chi(times[int(to[q])]), "entry_px": float(pin[q]), "exit_px": float(pout[q]),
               "size_usd": size, "cost": size, "lock": lock, "val": val, "fee_pct_side": pc,
               "fees_usd": size * pc * (1 + gross) + 2 * fix + K.RENT_UNRECOVERED * lock,
               "open": bool(isopen[q]), "reason": "open (marked at last close)" if isopen[q] else REASON.get(int(rs[q]), "?")}
        slots[slots.index(None)] = rec
    close_before(1 << 40)
    return cash, trades, skipped, maxdd

def run_strategy(D, cfg, start_ts):
    times = D["times"]; T = int(D["T"])
    t_lo = int(np.searchsorted(times, start_ts))  # first bar whose OPEN is at/after the live start
    row = cfg["row"]
    sig = wf_lib.signal_from_row(row, D, {})
    s = np.array(sig, copy=True)
    if t_lo > 1: s[: t_lo - 1] = False            # signal on bar t fills at bar t+1's open: nothing before start
    e = np.asarray(json.loads(row["exit_params"]), float)
    tr = E.gen_trades(s, e, D["O"], D["H"], D["L"], D["C"], D["last"], D["supply"], False)
    from solstrats import window as W
    tok, ti, to, pin, pout, isopen = W.clip_to_window(D, tr, t_lo, T) if t_lo < T else [np.zeros(0, int)] * 6
    rs = tr[5][(tr[1] >= t_lo) & (tr[1] < T)] if t_lo < T else np.zeros(0, int)
    bal = cfg["balance"]
    if t_lo < T:
        cash, trades, skipped, maxdd = sim_log(D, tok, ti, to, pin, pout, rs, isopen, t_lo, T, bal, cfg["frac"], cfg["maxpos"])
        chk = W.run(D, tr, t_lo, T, bal, cfg["frac"], cfg["maxpos"])   # cross-check against the numba engine
        if abs(float(chk[0]) - cash) > 1e-6 * max(1, abs(cash)): log(f"WARNING {cfg['id']}: python {cash} vs engine {chk[0]}")
    else:
        cash, trades, skipped, maxdd = bal, [], [], 0.0
    # pending: signal on the last complete bar -> buys at the next bar's open
    pending = [D["metas"][j].get("sym", "?") for j in np.nonzero(s[T - 1])[0]] if T >= 1 else []
    closed = [t for t in trades if not t["open"]]; opn = [t for t in trades if t["open"]]
    wins = sum(1 for t in closed if t["pnl"] > 0)
    return {"id": cfg["id"], "equity": cash, "ret_pct": (cash / bal - 1) * 100, "closed": closed, "open": opn,
            "n_closed": len(closed), "wins": wins, "win_rate": (wins / len(closed) * 100) if closed else None,
            "max_dd_pct": maxdd * 100, "skipped": skipped[-20:], "n_skipped": len(skipped), "pending": pending,
            "live_start_bar_time": chi(times[t_lo]) if t_lo < T else "waiting for first complete bar"}

def cycle(state):
    st = refresh()
    log(f"refresh: {st}")
    D = prep(str(DATA))
    D["metas"] = [json.loads(p.read_text())["meta"] for p in sorted((DATA / "ohlcv").glob("*.json"))
                  if len(json.loads(p.read_text()).get("ohlcv") or []) >= 3]
    assert len(D["metas"]) == D["N"], (len(D["metas"]), D["N"])
    out = []
    for cfg in state["strategies"]:
        try: out.append(run_strategy(D, cfg, state["start_ts"]))
        except Exception as ex: log(f"strategy {cfg['id']} failed: {ex!r}\n{traceback.format_exc()}")
    now = time.time()
    state.update(last_cycle=chi(now), last_cycle_ts=now, data_end=chi(D["times"][-1]), data_end_ts=float(D["times"][-1]),
                 refresh=st, results={r["id"]: {k: v for k, v in r.items()} for r in out}, cycles=state.get("cycles", 0) + 1)
    atomic_json(ROOT / "state.json", state)
    # trades.csv: full replayed log (closed + open), rewritten each cycle
    cols = ["strategy", "status", "sym", "entry_time", "exit_time", "entry_px", "exit_px", "size_usd", "pnl_usd", "pnl_pct",
            "fees_usd", "reason", "mint", "pool"]
    with open(ROOT / "trades.csv.tmp", "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in out:
            for t in r["closed"] + r["open"]:
                w.writerow([r["id"], "OPEN" if t["open"] else "CLOSED", t["sym"], t["entry_time"], "" if t["open"] else t["exit_time"],
                            f"{t['entry_px']:.10g}", f"{t['exit_px']:.10g}", f"{t['size_usd']:.4f}", f"{t['pnl']:.4f}",
                            f"{t['pnl'] / t['size_usd'] * 100:.2f}", f"{t['fees_usd']:.4f}", t["reason"], t["mint"], t["pool"]])
    os.replace(ROOT / "trades.csv.tmp", ROOT / "trades.csv")
    new = not (ROOT / "equity.csv").exists()
    with open(ROOT / "equity.csv", "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["time_chi", "ts", "data_end_chi", "strategy", "equity_usd", "ret_pct", "open_positions", "closed_trades"])
        for r in out:
            w.writerow([chi(now), int(now), chi(D["times"][-1]), r["id"], f"{r['equity']:.4f}", f"{r['ret_pct']:.2f}", len(r["open"]), r["n_closed"]])
    try:
        import live_panel; live_panel.inject()
    except Exception as ex: log(f"dashboard panel inject failed: {ex!r}")
    log("cycle done: " + " | ".join(f"{r['id']} ${r['equity']:.2f} ({len(r['open'])} open, {r['n_closed']} closed)" for r in out))

def main():
    ROOT.mkdir(exist_ok=True); init_data()
    sp = ROOT / "state.json"
    state = json.loads(sp.read_text())
    log(f"start/resume: live start {chi(state['start_ts'])}, strategies {[s['id'] for s in state['strategies']]}, pid {os.getpid()}")
    (ROOT / "trader.pid").write_text(str(os.getpid()))
    while True:
        try: cycle(state)
        except Exception as ex: log(f"cycle error: {ex!r}\n{traceback.format_exc()}")
        nxt = (int(time.time() // 3600) + 1) * 3600 + 150   # 2.5 min after the next hour closes
        time.sleep(max(30, nxt - time.time()))

if __name__ == "__main__":
    main()
