import json, time, requests
S=requests.Session(); S.headers['accept']='application/json'
EST="BONK WIF POPCAT MEW BOME GOAT FWOG MOODENG PNUT FARTCOIN TRUMP MELANIA PENGU AI16Z ZEREBRO CHILLGUY GIGA MICHI PONKE SLERF MYRO WEN SAMO RETARDIO SC LOCKIN ACT GRIFFAIN SWARMS ARC USELESS MOTHER BILLY GME DADDY HAWK LIBRA JAILSTOOL TROLL SPX PIPPIN ALCH UFD MOG NUB".split()
out=[]
for s in EST:
    try:
        r=S.get("https://api.dexscreener.com/latest/dex/search",params={"q":s},timeout=20).json()
        ps=[p for p in r.get('pairs') or [] if p['chainId']=='solana' and p['baseToken']['symbol'].upper()==s and p['quoteToken']['symbol'] in('SOL','USDC','USDT')]
        if not ps: print('none',s); continue
        p=max(ps,key=lambda p:(p.get('liquidity') or {}).get('usd',0))
        out.append(dict(sym=s,pool=p['pairAddress'],mint=p['baseToken']['address'],dex=p['dexId'],liq=(p.get('liquidity')or{}).get('usd'),mcap=p.get('marketCap'),created=p.get('pairCreatedAt'),group='established'))
    except Exception as e: print('err',s,e)
    time.sleep(0.4)
seen={o['mint'] for o in out}
cut=time.time()-175*86400
for dex in ['pumpswap','pump-fun','raydium-launchlab','letsbonk-fun']:
  for sort in ['h24_volume_usd_desc','h24_tx_count_desc']:
    for page in range(1,11):
        r=S.get(f"https://api.geckoterminal.com/api/v2/networks/solana/dexes/{dex}/pools",params={"page":page,"sort":sort,"include":"base_token"},timeout=20)
        time.sleep(2.2)
        if r.status_code!=200: print(dex,page,r.status_code); break
        d=r.json(); toks={t['id']:t['attributes'] for t in d.get('included',[])}
        if not d['data']: break
        for p in d['data']:
            a=p['attributes']; bt=p['relationships']['base_token']['data']['id']; t=toks.get(bt,{})
            mint=t.get('address'); sym=t.get('symbol','?')
            if mint in seen or sym in('SOL','WSOL','USDC'): continue
            seen.add(mint)
            out.append(dict(sym=sym,pool=a['address'],mint=mint,dex=dex,liq=float(a.get('reserve_in_usd') or 0),mcap=a.get('market_cap_usd') or a.get('fdv_usd'),created=a['pool_created_at'],group='launch'))
    print(dex,len(out))
json.dump(out,open('data/universe_raw.json','w'),indent=1); print(len(out))
