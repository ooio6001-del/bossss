import json, time, requests
S=requests.Session()
u=json.load(open('data/universe_raw.json'))
est=[x for x in u if x['group']=='established']
seen={x['mint'] for x in est}
KW="cat dog pepe trump ai moon inu elon frog based chad wojak doge shib bonk pump sol meme baby king god coin gpt agent bull bear monkey ape penguin hamster tiger rat duck fish girl boy wife mom dad grok musk president america china japan 2026 2025 lol wtf rug fart cum gm wagmi nigga retard autism pnut hippo goat wif cto stonk lambo rich poor money cash gold diamond rocket mars alien zombie ghost devil jesus pope bitcoin eth xrp lumpy giga mini little big super ultra degen".split()
cut=(time.time()-178*86400)*1000
launch=[]
for k in KW:
    try: r=S.get("https://api.dexscreener.com/latest/dex/search",params={"q":k},timeout=20).json()
    except Exception as e: print(e); continue
    for p in r.get('pairs') or []:
        if p['chainId']!='solana' or p['quoteToken']['symbol'] not in('SOL','USDC'): continue
        if p['dexId'] not in('pumpswap','raydium','meteora','pumpfun','launchlab'): continue
        c=p.get('pairCreatedAt') or 0
        m=p['baseToken']['address']
        if c<cut or m in seen: continue
        if (time.time()*1000-c)<7*86400*1000: continue  # need >=7d of history
        seen.add(m)
        launch.append(dict(sym=p['baseToken']['symbol'],pool=p['pairAddress'],mint=m,dex=p['dexId'],liq=(p.get('liquidity')or{}).get('usd',0),mcap=p.get('marketCap'),created=c,group='launch'))
    time.sleep(0.3)
print(len(launch))
json.dump(est+launch,open('data/universe.json','w'),indent=1)
import collections;print(collections.Counter(x['dex'] for x in launch))
liq=sorted(x['liq'] or 0 for x in launch);print(liq[:5],liq[len(liq)//2],liq[-5:])
