"""Month-by-month returns (America/Chicago calendar months, equity marked at cost at each close, from flat at the data start)
for the headline and the top stress survivors. -> results/monthly.json"""
import json, os, sys
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".numba_cache"))
sys.dont_write_bytecode = True
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from strategies import Registry
from compute_windows import strategy_rows
from solstrats import engine as E, costs as K
from run_backtest import prep
TZ = ZoneInfo("America/Chicago")

def curve(D, s, reg):
    tok, ti, to, pin, pout, _ = reg.trades(s, 0); T = int(D["T"]); cv = np.full(T, np.nan)
    ve = D["V"][np.maximum(ti - 1, 0), tok]
    m = E.simulate(tok, ti, to, pin, pout, D["venue"], ve, D["solpx"], 0, T, s["bal"], s["frac"], int(s["maxpos"]), K.PCT_COST, K.FIXED_SOL,
                   K.RENT_SOL, K.RENT_UNRECOVERED, K.IMPACT_K, K.FILL_CAP, cv)
    cv[0] = s["bal"] if np.isnan(cv[0]) else cv[0]
    for t in range(1, T):
        if np.isnan(cv[t]): cv[t] = cv[t - 1]
    return cv, float(m[1])

def main(R="results", n=6):
    D = prep(); reg = Registry(D, R); st = json.load(open(f"{R}/stress.json"))["rows"]
    ids = ["H6-01098"] + [r["id"] for r in st if r["survives"] and r["id"] != "H6-01098"][: n - 1]
    rows = {r["id"]: r for r in strategy_rows(R) if r["_tag"] == "robust"}
    months = [datetime.fromtimestamp(float(t), TZ).strftime("%Y-%m") for t in D["times"]]
    labels = sorted(set(months)); out = []
    for i in ids:
        s = reg.spec(rows[i]); cv, tot = curve(D, s, reg); rets = []; prev = s["bal"]
        for mth in labels:
            idx = max(k for k, m in enumerate(months) if m == mth); v = float(cv[idx]); rets.append(round(v / prev - 1, 4)); prev = v
        out.append({"id": i, "balance": s["bal"], "months": rets, "total": round(tot, 4), "up_months": sum(r > 0 for r in rets)})
        print(i, rets, tot)
    json.dump({"labels": labels, "partial": {"first": labels[0], "last": labels[-1]}, "rows": out}, open(f"{R}/monthly.json", "w"))

if __name__ == "__main__":
    main()
