import sys, csv
sys.argv=[sys.argv[0]]
import last48 as L
D=L.prep(str(L.DST_DATA)); L.warmup(D)
row=[r for r in csv.DictReader(open('robust_leaderboard.csv')) if r['id']=='H6-01098'][0]
t_lo,t_hi,_,_=L.window_bounds(D,4296)
w30=L.window_bounds(D,720)[:2]
for b in [5,10,15,20,25,30,40,50,60,75,100,125,150,175,200,250,280,500,1000]:
    r=dict(row); r['balance']=str(b)
    res,_=L.eval_hunt(D,[r],[(4296,t_lo,t_hi),(720,*w30)],{})
    f=res[4296][0]; m=res[720][0]
    print(b, round(f['end_bal'],2), round(f['ret_pct'],1), f['trades'], round(f['max_dd']*100), '| 30d', round(m['ret_pct'],1))
