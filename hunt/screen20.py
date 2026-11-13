import sys, csv, json, time
sys.argv=[sys.argv[0]]
import numpy as np
import last48 as L, wf_lib
E=L.E
D=L.prep(str(L.DST_DATA)); L.warmup(D)
T=int(D['T']); HO=336
print('T',T,L.chi(D['times'][0]),'->',L.chi(D['times'][T-1]),flush=True)
SIZ=[(1.0,1),(0.5,2),(0.34,3),(0.2,5)]
BAL=20.0
vol=lambda ti,tok: D['V'][np.maximum(ti-1,0),tok] if tok.size else np.zeros(0)
def sim(tr,lo,hi,fr,mp):
    tok,ti,to,pin,pout,_=tr
    return L.simulate_trades(D,tok,ti,to,pin,pout,lo,hi,fr,mp,BAL)
cands=[]  # (key, label dict, sig_fn, exit, one_shot, own_sizing)
board=list(csv.DictReader(open('leaderboard.csv')))+list(csv.DictReader(open('robust_leaderboard.csv')))
seen={}
for r in board:
    k=('H',r['batch'],r['family'],r['params'],r['exit_params'])
    if k not in seen: seen[k]=r
print('hunter combos',len(seen),flush=True)
# grid sweep combos at $5
z=np.load('/workspace/solstrats/results/sweep.npz',mmap_mode='r')
meta=json.load(open('/workspace/solstrats/results/sweep_meta.json'))
cb=np.asarray(z['cfg_bal']); cs=np.asarray(z['cfg_size']); cc=np.asarray(z['cfg_combo'])

sel=np.where((cb==20.0) & (cs<=4) & (cs>=1))[0]
fr=np.asarray(z['full'][:, 1])[sel]
good=sel[fr>0]
gcombos=sorted(set(cc[good][np.argsort(-fr[fr>0])][:20000].tolist()))
print('grid combos with +ret at $5',len(gcombos),flush=True)
caches={}; sigc={}
rows=[]
t0=time.time()
def evaluate(tag,label,tr,own):
    sizes=list(SIZ)+([own] if own and own not in SIZ else [])
    for fr_,mp in sizes:
        f=sim(tr,0,T,fr_,mp)
        if f[1]<=0: continue
        h=sim(tr,T-HO,T,fr_,mp); m30=sim(tr,T-720,T,fr_,mp)
        rows.append(dict(label,frac=fr_,maxpos=mp,end=float(f[0]),ret=float(f[1]),trades=int(f[4]),win=float(f[13]),dd=float(f[2]),
            r30=float(m30[1]),hret=float(h[1]),htr=int(h[4]),_tag=tag))
TR={}
for i,(k,r) in enumerate(seen.items()):
    sk=k[:4]
    if sk not in sigc:
        try: sigc[sk]=wf_lib.signal_from_row(r,D,caches)
        except Exception as ex: sigc[sk]=None
    if sigc[sk] is None: continue
    e=json.loads(r['exit_params'])
    tr=E.gen_trades(sigc[sk],np.asarray(e,float),D['O'],D['H'],D['L'],D['C'],D['last'],D['supply'],False)
    tag=('H',)+k[1:]; TR[tag]=tr
    try: own=wf_lib.parse_sizing(r['sizing'])
    except: own=None
    evaluate(tag,dict(id=r['id'],src='hunt',family=r['family'],entry=r['entry'],exit=r['exit']),tr,own)
    if i%500==0: print('hunt',i,time.time()-t0,len(rows),flush=True)
import run_backtest as RB
from solstrats.describe import entry_text, exit_text
gsig={}
for j,c in enumerate(gcombos):
    cm=meta['combos'][c]
    sk=(cm['family'],json.dumps(cm['p']))
    if sk not in gsig:
        fi=E.F[cm['family']]; p=cm['p']
        hot=D['hot'].get(int(p[0]),D['hot'][24]) if cm['family']=='narrative' else D['hot'][24]
        gsig[sk]=E.signals(fi,np.asarray(p,float),D['O'],D['H'],D['L'],D['C'],D['V'],D['first'],D['grp'],D['venue'],D['supply'],D['mint_auth'],D['freeze'],D['top10'],hot,D['theme'],0,T-1)
    tr=E.gen_trades(gsig[sk],np.asarray(cm['e'],float),D['O'],D['H'],D['L'],D['C'],D['last'],D['supply'],bool(cm['one_shot']))
    tag=('G',c); TR[tag]=tr
    try: ent=entry_text(cm['family'],cm['p'])
    except Exception: ent=cm['family']+str(cm['p'])
    try: ex=exit_text(cm['e'])
    except Exception: ex=str(cm['e'])
    evaluate(tag,dict(id=f'G{c}',src='grid',family=cm['family'],entry=ent,exit=ex),tr,None)
    if j%1000==0: print('grid',j,time.time()-t0,len(rows),flush=True)
print('evaluated rows',len(rows),flush=True)
ok=[r for r in rows if r['hret']>0 and r['htr']>0]
ok.sort(key=lambda r:-r['ret'])
print('profitable full & holdout',len(ok))
out=[]
for r in ok[:600]:
    tok,ti,to,pin,pout,_=TR[r['_tag']]
    folds,s=wf_lib.walk_forward_trades(D,tok,ti,to,pin,pout,r['frac'],r['maxpos'],BAL,end_bar=T-HO)
    a=s['aggregates']['all_folds']; ro=s['aggregates']['rolling']
    r.update(wf_n=a['n'],wf_pct=a.get('oos_profitable_pct'),wf_roll_n=ro['n'],wf_roll_pct=ro.get('oos_profitable_pct'),wf_med=a.get('oos_ret_median'),wf_pass=bool(s.get('passed')))
    out.append(r)
import pickle
pickle.dump({'rows':rows,'out':out},open('/workspace/solstrats_live/screen20.pkl','wb'))
keys=['id','src','family','frac','maxpos','end','ret','trades','win','dd','r30','hret','htr','wf_n','wf_pct','wf_roll_pct','wf_med','wf_pass','entry','exit']
with open('/workspace/solstrats_live/screen20_top.csv','w') as f:
    w=csv.DictWriter(f,keys,extrasaction='ignore'); w.writeheader(); [w.writerow(r) for r in out]
print('done',time.time()-t0)
