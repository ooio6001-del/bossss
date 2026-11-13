"""Random-entry null test: keep each strategy's exits, sizing, balance and the number of signals per token, but move every signal
to a random hour in that token's trading life (300 draws). p = share of random draws that match or beat the real full-period return.
A low p says the entry rule adds something beyond the exit + sizing + token universe; it does not correct for how many rules were
searched to find it. -> results/null_test.json"""
from __future__ import annotations
import json, os, sys
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".numba_cache"))
sys.dont_write_bytecode = True
import numpy as np
from strategies import Registry
from compute_windows import strategy_rows
from solstrats import engine as E
from run_backtest import prep
from stress import sim

DRAWS = int(os.environ.get("NULL_DRAWS", "300"))

def null_one(D, reg, s, rng):
    sig = reg.signal(s); e = np.asarray(s["e"], float); first, last = D["first"], D["last"]
    real = sim(D, reg.trades(s, 0), s["bal"], s["frac"], s["maxpos"])[0]
    cols = [(j, int(sig[:, j].sum())) for j in range(sig.shape[1]) if sig[:, j].any() and last[j] > first[j] + 1]
    rets = np.empty(DRAWS)
    for d in range(DRAWS):
        rs = np.zeros_like(sig)
        for j, k in cols:
            lo, hi = max(int(first[j]), 0), int(last[j])
            rs[rng.choice(np.arange(lo, hi), size=min(k, hi - lo), replace=False), j] = True
        rets[d] = sim(D, E.gen_trades(rs, e, D["O"], D["H"], D["L"], D["C"], D["last"], D["supply"], bool(s["one_shot"])), s["bal"], s["frac"], s["maxpos"])[0]
    return {"id": s["id"], "family": s["family"], "real_ret": real, "null_median": float(np.median(rets)), "null_p95": float(np.percentile(rets, 95)),
            "null_profitable": float((rets > 0).mean()), "p_value": float(((rets >= real).sum() + 1) / (DRAWS + 1)), "draws": DRAWS}

def main(R="results"):
    D = prep(); reg = Registry(D, R); rng = np.random.default_rng(11)
    st = json.load(open(f"{R}/stress.json"))["rows"]
    want = [r["id"] for r in st]
    rows = {r["id"]: r for r in strategy_rows(R) if r["_tag"] == "robust"}
    out, done = [], set()
    for i in want:
        if i in done or i not in rows: continue
        done.add(i); s = reg.spec(rows[i])
        if s: out.append(null_one(D, reg, s, rng)); print(out[-1], flush=True)
    json.dump({"draws": DRAWS, "rows": out}, open(f"{R}/null_test.json", "w"))

if __name__ == "__main__":
    main()
