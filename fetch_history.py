"""Download hourly OHLCV (GeckoTerminal public API, last 180 days max) for the token universe into data/ohlcv/.
Established coins: the highest-volume pool older than the window; all pages back to 179 days.
Launches: the first 1000 hours after the pool was created (the part of a launch's life the launch families trade).
Throttled to the GeckoTerminal public limit with exponential backoff + jitter (solstrats.http.Client)."""
import json, logging, os, random, sys, time
from solstrats.http import Client, HttpError
GT = "https://api.geckoterminal.com/api/v2/networks/solana"
logging.basicConfig(level=logging.WARNING)

def main(n_launch=100):
    c = Client(cache_dir="data/cache/gt")
    u = json.load(open("data/universe.json"))
    est = [x for x in u if x["group"] == "established"]
    la = [x for x in u if x["group"] == "launch" and (x.get("liq") or 0) < 5e7]
    random.seed(7); random.shuffle(la)
    todo = [x for x in u if x["group"] == "benchmark"] + la[:n_launch] + est
    floor = time.time() - 179 * 86400
    os.makedirs("data/ohlcv", exist_ok=True); fails = []
    for x in todo:
        if x["group"] == "established" and not x.get("pool_checked"):
            try:
                d = c.get("geckoterminal", f"{GT}/tokens/{x['mint']}/pools", params={"sort": "h24_volume_usd_desc"})
                vol = lambda q: float((q["attributes"].get("volume_usd") or {}).get("h24") or 0)
                ps = sorted([p for p in d["data"] if p["relationships"]["base_token"]["data"]["id"].endswith(x["mint"])], key=vol, reverse=True)
                old = [q for q in ps[:6] if vol(q) > 0.2 * vol(ps[0]) and q["attributes"]["pool_created_at"] < time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(floor))]
                if old or ps:
                    x["pool"] = (old or ps)[0]["attributes"]["address"]
            except (HttpError, KeyError, IndexError) as e:
                fails.append((x["sym"], "pool-select", str(e)))
        fn = f"data/ohlcv/{x['pool']}.json"
        if os.path.exists(fn): continue
        rows, before = [], None
        try:
            while True:
                p = {"aggregate": 1, "limit": 1000, "currency": "usd", "token": "base"}
                if before: p["before_timestamp"] = int(before)
                elif x["group"] == "launch" and isinstance(x.get("created"), (int, float)):
                    p["before_timestamp"] = int(min(time.time(), x["created"] / 1000 + 1000 * 3600))
                d = c.get("geckoterminal", f"{GT}/pools/{x['pool']}/ohlcv/hour", params=p)
                l = d["data"]["attributes"]["ohlcv_list"]; rows += l
                if len(l) < 1000 or x["group"] == "launch": break
                before = min(r[0] for r in l)
                if before < floor + 3600: break
        except (HttpError, KeyError) as e:
            if not rows: fails.append((x["sym"], x["pool"], str(e))); continue
        rows = sorted({r[0]: r for r in rows}.values())
        json.dump({"meta": x, "ohlcv": rows}, open(fn, "w"))
        print(x["sym"], len(rows), c.stats, flush=True)
    json.dump(fails, open("data/fetch_failures.json", "w"), indent=1)
    print("DONE", c.stats, "failures", len(fails))

if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:2]])
