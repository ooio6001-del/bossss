"""365-day DAILY history for established coins via CoinGecko public API (GeckoTerminal public caps at 180 days)."""
import json, os, time, logging
from solstrats.http import Client, Limit
logging.basicConfig(level=logging.WARNING)
c = Client(cache_dir='data/cache/cg', limits={"coingecko": Limit(8, 1)}, base_delay=10)
u = json.load(open('data/universe.json'))
lst = c.get("coingecko", "https://api.coingecko.com/api/v3/coins/list", params={"include_platform": "true"}, cache_ttl=86400*7)
by_mint = {}
for co in lst:
    m = (co.get("platforms") or {}).get("solana")
    if m: by_mint[m] = co["id"]
by_mint["So11111111111111111111111111111111111111112"] = "solana"
os.makedirs('data/daily', exist_ok=True)
for x in u:
    if x['group'] == 'launch': continue
    cid = by_mint.get(x['mint'])
    if not cid: print('no cg id', x['sym']); continue
    fn = f"data/daily/{x['sym']}.json"
    if os.path.exists(fn): continue
    try:
        d = c.get("coingecko", f"https://api.coingecko.com/api/v3/coins/{cid}/market_chart", params={"vs_currency": "usd", "days": 365, "interval": "daily"})
        json.dump({"sym": x['sym'], "id": cid, "prices": d["prices"], "vols": d["total_volumes"], "mcaps": d["market_caps"]}, open(fn, 'w'))
        print(x['sym'], len(d['prices']), flush=True)
    except Exception as e:
        print('fail', x['sym'], type(e).__name__, getattr(e, 'status', ''), flush=True)
print('DONE', c.stats)
