import sys,csv,json,pickle
sys.argv=[sys.argv[0]]; sys.path.insert(0,'/workspace/solstrats_hunt')
import numpy as np, last48 as L, wf_lib
D=L.prep(str(L.DST_DATA)); L.warmup(D); T=int(D['T'])
d=pickle.load(open('/workspace/solstrats_live/screen20.pkl','rb'))
pick=[('H7-06668',0.29,3),('G9658',0.34,3),('H7-04923',0.34,3),('H7-07282',0.2,5),('H7-04028',0.5,2)]
out=[]
for i,f,m in pick:
    r=[x for x in d['out'] if x['id']==i and x['frac']==f and x['maxpos']==m][0]; out.append(r)
# H6-01098 at $20 own sizing
row=[r for r in csv.DictReader(open('robust_leaderboard.csv')) if r['id']=='H6-01098'][0]
sig=wf_lib.signal_from_row(row,D,{}); e=json.loads(row['exit_params'])
tr=L.E.gen_trades(sig,np.asarray(e,float),D['O'],D['H'],D['L'],D['C'],D['last'],D['supply'],False)
fr,mp=wf_lib.parse_sizing(row['sizing'])
S=lambda lo:L.simulate_trades(D,*tr[:5],lo,T,fr,mp,20.0)
f=S(0);h=S(T-336);m30=S(T-720)
folds,s=wf_lib.walk_forward_trades(D,*tr[:5],fr,mp,20.0,end_bar=T-336)
out.append(dict(id='H6-01098',family='washout_gate',frac=fr,maxpos=mp,end=float(f[0]),ret=float(f[1]),trades=int(f[4]),win=float(f[13]),dd=float(f[2]),r30=float(m30[1]),hret=float(h[1]),htr=int(h[4]),wf_n=s['aggregates']['all_folds']['n'],wf_pct=s['aggregates']['all_folds']['oos_profitable_pct'],wf_roll_pct=s['aggregates']['rolling']['oos_profitable_pct'],wf_pass=s['passed'],entry=row['entry'],exit=row['exit']))
for r in out:
    print(f"{r['id']:9} {r['family']:12} {r['frac']*100:.0f}%x{r['maxpos']} end ${r['end']:.2f} ret {r['ret']*100:+.1f}% tr {r['trades']} win {r['win']*100:.1f}% dd {r['dd']*100:.1f}% 30d {r['r30']*100:+.1f}% ho {r['hret']*100:+.2f}% ({r['htr']} tr) folds {r['wf_pct']}% all / {r['wf_roll_pct']}% roll of {r['wf_n']} pass={r['wf_pass']}")
    print('    ',r['entry'],'|',r['exit'])
json.dump([{k:v for k,v in r.items() if k!='_tag'} for r in out],open('/workspace/solstrats_live/top5_at20.json','w'),indent=1,default=str)
