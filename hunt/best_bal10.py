import sys, csv
sys.argv=[sys.argv[0]]
import last48 as L
D=L.prep(str(L.DST_DATA)); L.warmup(D)
rob=list(csv.DictReader(open('robust_leaderboard.csv')))
row=[r for r in rob if r['id']=='H6-01098'][0]
print('orig bal',row['balance'],row['sizing'])
T=D['T'] if isinstance(D,dict) and 'T' in D else None
out=[]
for h in (24*7, 24*30, 24*90, 4296):
    t_lo,t_hi,ts0,ts1=L.window_bounds(D,h)
    for b in ('10',row['balance']):
        r=dict(row); r['balance']=b
        res,err=L.eval_hunt(D,[r],[(h,t_lo,t_hi)],{})
        x=res[h][0]
        print(h,'h',L.chi(ts0),'->',L.chi(ts1),'bal',b,{k:x.get(k) for k in ('end_bal','profit','ret_pct','trades','wins','max_dd','fees','skipped')}, 'err',err)
