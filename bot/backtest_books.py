#!/usr/bin/env python3
"""Backtest every book in books.json on the candles this bot has downloaded (run/data), same engine + cost model.
usage: python backtest_books.py [hours]   (default: all downloaded history; e.g. 720 = last 30 days)"""
import json, os, sys
from pathlib import Path
HOME = Path(os.environ.get("SOLSTRATS_HOME") or Path(__file__).resolve().parent)
RUN = Path(os.environ.get("SOLSTRATS_RUN_DIR") or HOME / "run")
sys.path.insert(0, str(HOME / "lib"))
import numpy as np
from solstrats import costs as K, engine as E
from run_backtest import prep
import wf_lib
D = prep(str(RUN / "data")); T = int(D["T"])
hours = int(sys.argv[1]) if len(sys.argv) > 1 else T
lo = max(0, T - hours)
print(f"window: last {T - lo} hourly bars")
for b in json.loads((HOME / "books.json").read_text())["books"]:
    sig = np.array(wf_lib.signal_from_row(b["row"], D, {}), copy=True)
    if lo > 1: sig[: lo - 1] = False
    tr = E.gen_trades(sig, np.asarray(json.loads(b["row"]["exit_params"]), float), D["O"], D["H"], D["L"], D["C"], D["last"], D["supply"], False)
    ve = D["V"][np.maximum(tr[1] - 1, 0), tr[0]] if tr[0].size else np.zeros(0)
    m = E.simulate(tr[0], tr[1], tr[2], tr[3], tr[4], D["venue"], ve, D["solpx"], lo, T, float(b["balance"]), float(b["frac"]), int(b["maxpos"]),
                   K.PCT_COST, K.FIXED_SOL, K.RENT_SOL, K.RENT_UNRECOVERED, K.IMPACT_K, K.FILL_CAP, np.zeros(0))
    print(f"{b['label'][:46]:46} start ${b['balance']:>7.2f} -> ${m[0]:>9.2f} ({m[1]*100:+7.1f}%)  trades {int(m[4]):>4}  win {m[13]*100:5.1f}%  maxDD {m[2]*100:5.1f}%")
