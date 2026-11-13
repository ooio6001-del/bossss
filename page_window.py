"""Compact window data for the Page: exact preset results for a short list of strategies plus, for the last 7 days,
every candidate trade and hourly marks so the Page can recompute any hour-precision custom window inside that span."""
from __future__ import annotations
import json, sys
import numpy as np
from strategies import Registry
from compute_windows import strategy_rows
from solstrats import window as W
from run_backtest import prep

def main(R="results", n_robust=8, n_top=6, span=48, extra=(("robust", 12), ("top100", 10), ("top100_profit", 8))):
    D = prep(); reg = Registry(D, R); T = int(D["T"]); base = T - span
    rows = strategy_rows(R); pick, seen = [], set()
    for tag, n in (("robust", n_robust), ("top100", n_top), ("top100_profit", 2)):
        for r in rows:
            if r["_tag"] == tag and sum(p["_tag"] == tag for p in pick) < n:
                s = reg.spec(r)
                if s and s["id"] not in seen: s["_tag"] = tag; s["entry"] = r.get("entry", ""); s["exit"] = r.get("exit", ""); pick.append(s); seen.add(s["id"])
    n_custom = len(pick)
    for tag, n in extra:
        for r in rows:
            if r["_tag"] == tag and sum(p["_tag"] == tag for p in pick) < n:
                s = reg.spec(r)
                if s and s["id"] not in seen: s["_tag"] = tag; s["entry"] = r.get("entry", ""); s["exit"] = r.get("exit", ""); pick.append(s); seen.add(s["id"])
    toks = {}; strat = []
    for si, s in enumerate(pick):
        c = reg.candidates(s); eo = W.end_open(D, c); m = (c[1] >= base) & (si < n_custom)
        cand = []
        for q in np.nonzero(m)[0]:
            j = int(c[0][q]); toks.setdefault(j, len(toks))
            ve = float(D["V"][max(c[1][q] - 1, 0), j])
            cand.append(f"{toks[j]},{int(c[1][q]) - base},{int(c[2][q]) - base},{c[4][q] / c[3][q]:.4f},{c[3][q]:.5g},{ve:.3g},{int(eo[q])}")
        pres = {}
        for p in W.PRESETS_H.keys() | {"ytd", "12m"}:
            b = W.resolve(D, preset=p)
            mm = W.pack(W.run(D, W.replay(c, b["t_lo"], s["one_shot"]), b["t_lo"], b["t_hi"], s["bal"], s["frac"], s["maxpos"]), s["bal"])
            pres[p] = [round(mm["final"], 2), mm["ntrades"], round(mm["winrate"], 3), round(mm["maxdd"], 3), mm["n_open"], round(mm["unrealized"], 2)]
        strat.append({"id": s["id"], "tag": s["_tag"], "fam": s["family"], "rule": f'{s["entry"]} · {s["exit"]} · {s["sizing"]}'[:150],
                      "bal": s["bal"], "frac": s["frac"], "mp": s["maxpos"], "one": int(bool(s["one_shot"])), "cx": int(si < n_custom), "c": ";".join(cand), "p": pres})
    inv = sorted(toks, key=toks.get)
    marks = [[float(f"{D["C"][t, j]:.5g}") if np.isfinite(D["C"][t, j]) else 0 for t in range(base, T)] for j in inv]
    out = {"t0": float(D["times"][base]), "span": span, "venue": [int(D["venue"][j]) for j in inv], "sol": [round(float(x), 2) for x in D["solpx"][base:T]],
           "marks": marks, "strats": strat, "bounds": {p: W.resolve(D, preset=p) for p in list(W.PRESETS_H) + ["ytd", "12m"]}}
    json.dump(out, open(f"{R}/page_window.json", "w"), separators=(",", ":"))
    print("strategies", len(strat), "tokens", len(inv), "candidates", sum(len(s["c"].split(";")) for s in strat), "bytes", len(json.dumps(out, separators=(",", ":"))))

if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["results"]))
