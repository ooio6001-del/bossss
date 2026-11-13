"""Run the live paper trader for the top backtested strategies.
usage: python run_live.py [minutes=5] [interval_s=20] [balances=100,280]
Reads results/report.json for strategies; writes data/live/snapshot.json and data/live/log.jsonl."""
import json, logging, os, sys
RES = os.environ.get("SOLSTRATS_RESULTS", "results")
from solstrats.http import Client
from solstrats.providers import Providers
from solstrats.live import LiveRunner

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s %(message)s")

def strategies_from_report(n=6):
    rep = json.load(open(f"{RES}/report.json")); out = []
    meta = json.load(open(f"{RES}/sweep_meta.json"))
    picks = [r for r in rep["top100"] if r["id"].startswith("G")][:n]   # hunter rows use families outside this engine
    # always include the best launch-family strategies (they can fire live within minutes)
    for fam in ("desk", "moonshot", "fast_mc", "graduation"):
        f = [x for x in rep["families"] if x["family"] == fam and x["best"]]
        if f: picks.append(f[0]["best"])
    seen = set()
    for r in picks:
        if r["id"] in seen: continue
        seen.add(r["id"])
        g, s = r["id"][1:].split("-S"); cb = meta["combos"][int(g)]; fr, mp = meta["sizing"][int(s)]
        out.append({"id": r["id"], "family": cb["family"], "p": cb["p"], "e": cb["e"], "frac": fr, "maxpos": mp, "ovmask": 0,
                    "label": r["entry"], "bt_full_ret": r.get("full_ret"), "bt_oos_ret": r.get("oos_ret"), "bt_win": r.get("full_win"),
                    "bt_dd": r.get("full_dd")})
    return out

if __name__ == "__main__":
    minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 5
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 20
    bals = [float(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "100,280").split(",")]
    strats = strategies_from_report()
    u = json.load(open("data/universe.json"))
    watch = [{"mint": x["mint"], "symbol": x["sym"], "pool": x["pool"]} for x in u if x["group"] == "established"]
    pv = Providers(Client(cache_dir="data/live/cache"))
    lr = LiveRunner(strats, bals, watch=watch, providers=pv)
    os.makedirs("data/live", exist_ok=True)
    json.dump(strats, open("data/live/strategies.json", "w"), indent=1)
    lr.run(minutes * 60, interval)
    print("polls", lr.polls, "errors", lr.errors, "http", pv.c.stats, "provider failures", pv.status()["failures"])
