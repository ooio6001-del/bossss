import sys, pandas as pd
sys.argv=[sys.argv[0]]
import last48 as L
D=L.prep(str(L.DST_DATA)); L.warmup(D)
t_lo,t_hi,ts0,ts1=L.window_bounds(D,24)
print('window',L.chi(ts0),'->',L.chi(ts1))
rows,robust,rids,extra=L.load_hunt_rows()
seen=set();R=[]
for r in rows:
    k=(r['id'] .split('-')[0], r['family'], r['params'], r['exit_params'], r['sizing'])
    if k in seen: continue
    seen.add(k); r=dict(r); r['balance']='10'; R.append(r)
print('unique rows',len(R))
res,err=L.eval_hunt(D,R,[(24,t_lo,t_hi)],{})
df=pd.DataFrame(res[24]); df['robust']=df.id.isin(rids)
df.to_csv('last24_bal10.csv',index=False)
cols=[c for c in ['id','family','profit','ret_pct','end_bal','trades','wins','fees','entry','exit','sizing','robust'] if c in df]
t=df[df.trades>0].sort_values('profit',ascending=False)
print('traded',len(t),'profitable',(t.profit>0).sum(),'errors',len(err))
pd.set_option('display.max_colwidth',120)
print(t[cols].head(5).to_string())
print(t[t.robust][cols].head(3).to_string())
