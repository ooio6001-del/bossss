import sys,json,pickle,csv
sys.argv=['x']; sys.path.insert(0,'/workspace/solstrats_hunt')
import numpy as np, last48 as L, wf_lib
E=L.E
D=L.prep(str(L.DST_DATA)); L.warmup(D); T=int(D['T'])
d=pickle.load(open('screen20.pkl','rb')); out=d['out']
meta=json.load(open('/workspace/solstrats/results/sweep_meta.json'))
board={}
for r in list(csv.DictReader(open('/workspace/solstrats_hunt/leaderboard.csv')))+list(csv.DictReader(open('/workspace/solstrats_hunt/robust_leaderboard.csv'))):
    board.setdefault(('H',)+(r['batch'],r['family'],r['params'],r['exit_params']),r)
sc={}; caches={}
for r in out:
    tag=r['_tag']
    for lo,key in ((T-336,'hflat'),(T-720,'r30flat')):
        if tag[0]=='H':
            row=board[tag]; sk=tag[:4]
            if sk not in sc: sc[sk]=wf_lib.signal_from_row(row,D,caches)
            sig=sc[sk]; e=json.loads(row['exit_params']); one=False
        else:
            cm=meta['combos'][tag[1]]; sk=('G',cm['family'],json.dumps(cm['p']))
            if sk not in sc:
                p=cm['p']; hot=D['hot'][24]
                sc[sk]=E.signals(E.F[cm['family']],np.asarray(p,float),D['O'],D['H'],D['L'],D['C'],D['V'],D['first'],D['grp'],D['venue'],D['supply'],D['mint_auth'],D['freeze'],D['top10'],hot,D['theme'],0,T-1)
            sig=sc[sk]; e=cm['e']; one=bool(cm['one_shot'])
        tr=L.gen_window_trades(sig,e,D,lo,one)
        m=L.simulate_trades(D,*tr[:5],lo,T,r['frac'],r['maxpos'],20.0)
        r[key]=float(m[1]); r[key+'_tr']=int(m[4])
pickle.dump(d,open('screen20.pkl','wb'))
g=[r for r in out if r['wf_pct']>50 and r['wf_roll_pct']>50 and r['hret']>0 and r['hflat']>0]
seen=set()
for r in g:
    if r['id'] in seen: continue
    seen.add(r['id'])
    print(r['id'],r['family'],r['frac'],r['maxpos'],round(r['end'],2),round(r['ret']*100),r['trades'],round(r['win']*100,1),round(r['dd']*100,1),'30d',round(r['r30']*100,1),round(r['r30flat']*100,1),'ho',round(r['hret']*100,1),round(r['hflat']*100,1),r['hflat_tr'],'wf',r['wf_pct'],r['wf_roll_pct'])
    if len(seen)>=14: break
