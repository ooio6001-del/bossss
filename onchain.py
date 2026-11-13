"""Snapshot on-chain safety data (mint/freeze authority, holder concentration) for the universe.
Uses Helius RPC when HELIUS_API_KEY is set, else public Solana RPC. Key never printed."""
import json, logging, sys
from solstrats.http import Client
from solstrats.providers import Providers
logging.basicConfig(level=logging.WARNING)
u = json.load(open('data/universe.json'))
pv = Providers(Client(cache_dir='data/cache/rpc'))
out = {}
for x in u:
    if x['group'] == 'benchmark': continue
    try:
        out[x['mint']] = pv.mint_safety(x['mint'])
    except Exception as e:  # noqa: BLE001
        out[x['mint']] = {"error": type(e).__name__}
    r = out[x['mint']]
    print(x['sym'], {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != 'mint'}, flush=True)
json.dump(out, open('data/onchain.json', 'w'), indent=1)
print('DONE', pv.status()["failures"], pv.c.stats)
