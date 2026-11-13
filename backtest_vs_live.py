"""Backtest vs Live: for each live paper book (@$100), the same strategy's backtest on the last 24h window and on the
full hourly span (engine, from flat, same costs), next to the live paper result. -> results/backtest_vs_live.json"""
import json, os, sys
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".numba_cache"))
sys.dont_write_bytecode = True
from strategies import Registry
from solstrats import window as W
from run_backtest import prep

def main(R="results", bal=100.0):
    s = json.load(open("data/live/snapshot.json")); D = prep(); reg = Registry(D, R); out = []
    for k, b in s["books"].items():
        sid, b0 = k.split("@")
        if float(b0) != bal: continue
        sp = reg.spec({"id": sid, "balance": bal})
        row = {"book": k, "id": sid, "live_equity": round(b["equity"], 2), "live_ret": round(b["equity"] / b["balance0"] - 1, 4),
               "live_liq_ret": round(b["equity_liq"] / b["balance0"] - 1, 4) if "equity_liq" in b else None,
               "live_closed": len(b.get("closed", [])), "live_open": len(b.get("open", [])),
               "live_curve": [round(x[1] if isinstance(x, list) else x, 2) for x in (b.get("curve") or [])][-60:]}
        if sp:
            row["family"] = sp["family"]
            for p in ("24h", "12m"):
                bd = W.resolve(D, preset=p)
                m = W.pack(W.run(D, W.replay(reg.candidates(sp), bd["t_lo"], sp["one_shot"]), bd["t_lo"], bd["t_hi"], bal, sp["frac"], sp["maxpos"]), bal)
                row[f"bt_{p}_ret"] = round(m["ret"], 4); row[f"bt_{p}_trades"] = m["ntrades"] + m["n_open"]
        out.append(row)
    meta = {"started": s["started"], "ts": s["ts"], "polls": s["polls"], "errors": s["errors"], "http": s["http_stats"]}
    json.dump({"meta": meta, "rows": out}, open(f"{R}/backtest_vs_live.json", "w"))
    for r in out: print({k: v for k, v in r.items() if k != "live_curve"})
    print(meta)

if __name__ == "__main__":
    main()
