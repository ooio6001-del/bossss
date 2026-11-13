import json, time, requests, os, random
S=requests.Session(); S.headers['accept']='application/json'
u=json.load(open('data/universe.json'))
est=[x for x in u if x['group']=='established']
la=[x for x in u if x['group']=='launch' and (x['liq'] or 0)<5e7]
random.seed(7); random.shuffle(la)
todo=[x for x in u if x['group']=='benchmark']+la[:100]+est
os.makedirs('data/ohlcv',exist_ok=True)
floor=time.time()-179*86400
calls=0;t0=time.time();fails=[]
def get(url,params):
    global calls
    for att in range(10):
        r=S.get(url,params=params,timeout=30); calls+=1
        if r.status_code==429: time.sleep(15*(att+1)); continue
        time.sleep(4.0); return r
    return r
for i,x in enumerate(todo):
    if x['group']=='established':
        r=get(f"https://api.geckoterminal.com/api/v2/networks/solana/tokens/{x['mint']}/pools",{"sort":"h24_volume_usd_desc"})
        print('poolsel',x['sym'],r.status_code,flush=True)
        if r.status_code==200:
            ps=[p for p in r.json()['data'] if p['relationships']['base_token']['data']['id'].endswith(x['mint'])]
            vol=lambda q: float((q['attributes'].get('volume_usd') or {}).get('h24') or 0)
            ps=sorted(ps,key=vol,reverse=True)
            print('  top',[(q['attributes']['address'][:6],round(vol(q))) for q in ps[:4]],flush=True)
            old=[q for q in ps[:6] if vol(q)>0.2*vol(ps[0]) and q['attributes']['pool_created_at']<time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(floor))]
            ps=old or ps
            if ps:
                p=ps[0]['attributes']; x.update(pool=p['address'],liq=float(p.get('reserve_in_usd') or 0),vol24=float((p.get('volume_usd') or {}).get('h24') or 0),pool_created=p.get('pool_created_at'))
    fn=f"data/ohlcv/{x['pool']}.json"
    if os.path.exists(fn): continue
    rows=[]; before=None; err=None
    while True:
        p={"aggregate":1,"limit":1000,"currency":"usd","token":"base"}
        if before: p["before_timestamp"]=int(before)
        elif x['group']=='launch':
            cr=x['created']/1000 if isinstance(x['created'],(int,float)) else floor
            p["before_timestamp"]=int(min(time.time(),cr+1000*3600))
        r=get(f"https://api.geckoterminal.com/api/v2/networks/solana/pools/{x['pool']}/ohlcv/hour",p)
        if r.status_code!=200: err=f"{r.status_code}"; break
        l=r.json()['data']['attributes']['ohlcv_list']
        rows+=l
        if len(l)<1000 or x['group']=='launch': break
        before=min(c[0] for c in l)
        if before-1000*3600<floor: 
            before=before  # one more allowed only if within window
            if before<floor+3600: break
    if err and not rows: fails.append((x['sym'],x['pool'],err)); print('FAIL',x['sym'],err,flush=True); continue
    rows=sorted({c[0]:c for c in rows}.values())
    json.dump(dict(meta=x,ohlcv=rows),open(fn,'w'))
    print(i,x['sym'],len(rows),f"calls={calls} rate={calls/(time.time()-t0)*60:.1f}/min",flush=True)
json.dump(fails,open('data/fetch_failures.json','w'))
print('DONE',calls)
