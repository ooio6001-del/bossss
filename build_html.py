"""Standalone dark-terminal HTML dashboard (dashboard.html). Embeds results/report.json; when served with
`python -m http.server` from the project root it also polls data/live/snapshot.json every 15 s (no keys in the page)."""
import json, os, sys
R = sys.argv[1] if len(sys.argv) > 1 else "results"
rep = json.load(open(f"{R}/report.json"))
for _k, _f in (("micro_be", "micro_breakeven.json"), ("bt_live", "backtest_vs_live.json"), ("stress", "stress.json"), ("nullt", "null_test.json"), ("plateau", "plateau.json"), ("monthly", "monthly.json")):
    if os.path.exists(f"{R}/{_f}"):
        rep[_k] = json.load(open(f"{R}/{_f}"))
snap = {}
if os.path.exists("data/live/snapshot.json"):
    try:
        snap = json.load(open("data/live/snapshot.json"))
    except ValueError:
        snap = {}
HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>SOLSTRATS // terminal</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>#lstrip{padding:10px 12px 6px;border-bottom:1px solid var(--line);background:radial-gradient(1200px 120px at 20% 0,#14f19514,transparent),#05070d}#lstrip .lsh{font-size:11px;letter-spacing:.12em;color:#9fb3c8;margin-bottom:6px}#lstrip .lsh b{color:var(--g);text-shadow:0 0 8px #14f195aa}#lslab{color:#00E5FF;margin:0 8px}.lsrow{display:flex;gap:10px;overflow-x:auto;padding:4px 2px 10px;scroll-snap-type:x proximity;scrollbar-width:thin}.lsc{flex:0 0 172px;scroll-snap-align:start;background:linear-gradient(180deg,#0d1322,#080b14);border:1px solid #1c2338;border-radius:12px;padding:9px 11px;font-size:12px;position:relative}.lsc.r1{border-color:var(--g);box-shadow:0 0 18px #14f19566,inset 0 0 12px #14f19522}.lsc.rob{border-color:#14f19577}.lsc .rk{font-size:20px;font-weight:700;color:#00E5FF;text-shadow:0 0 10px #00e5ff88}.lsc .nm{color:var(--t);font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.lsc .fm{color:#7d8aa5;font-size:10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.lsc .pf{font-size:16px;font-weight:700;margin-top:4px}.lsc .up{color:#39FF14;text-shadow:0 0 8px #39ff1488}.lsc .dn{color:#FF2D95;text-shadow:0 0 8px #ff2d9588}.lsc .mt{display:flex;justify-content:space-between;color:#9fb3c8;font-size:11px;margin-top:2px}.lsc .bd{position:absolute;top:8px;right:8px;font-size:9px;letter-spacing:.08em;padding:2px 6px;border-radius:999px;border:1px solid}.lsc .bd.y{color:#39FF14;border-color:#39FF14;box-shadow:0 0 8px #39ff1455}.lsc .bd.n{color:#7d8aa5;border-color:#2a3350}#jump{position:sticky;top:0;z-index:20;display:flex;gap:6px;overflow-x:auto;padding:8px 12px;background:#05070dee;border-bottom:1px solid var(--line);backdrop-filter:blur(6px);scrollbar-width:thin}#jump a{white-space:nowrap;font-size:11px;color:#9fb3c8;text-decoration:none;border:1px solid #1c2338;border-radius:999px;padding:3px 9px}#jump a:hover{color:#39FF14;border-color:#39FF14}section.card{scroll-margin-top:48px}
:root{--bg:#05060a;--panel:#0b0e17;--line:#1b2133;--g:#14f195;--p:#9945ff;--c:#00d1ff;--r:#ff4d6d;--t:#d7def0;--m:#7c86a2}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(1200px 600px at 10% -10%,#1a0f33 0,transparent 60%),radial-gradient(900px 500px at 110% 0,#06281f 0,transparent 55%),var(--bg);color:var(--t);font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
header{display:flex;align-items:center;gap:16px;padding:14px 22px;border-bottom:1px solid var(--line);position:sticky;top:0;background:#05060ae6;backdrop-filter:blur(8px);z-index:5}
.logo{font-weight:800;letter-spacing:.18em;background:linear-gradient(90deg,var(--g),var(--p));-webkit-background-clip:text;color:transparent;font-size:18px;text-shadow:0 0 24px #14f19555}
.ticker{flex:1;overflow:hidden;white-space:nowrap;mask-image:linear-gradient(90deg,transparent,#000 8%,#000 92%,transparent)}
.ticker span{display:inline-block;padding-right:40px;animation:tk 60s linear infinite}.ticker b{color:var(--g)}.ticker i{color:var(--r);font-style:normal}
@keyframes tk{from{transform:translateX(0)}to{transform:translateX(-50%)}}
.dot{width:8px;height:8px;border-radius:50%;background:var(--g);box-shadow:0 0 10px var(--g);animation:pl 1.6s infinite}@keyframes pl{50%{opacity:.3}}
main{padding:18px 22px;display:grid;gap:16px;grid-template-columns:repeat(12,1fr)}
.card{grid-column:span 12;background:linear-gradient(180deg,#0d1120,#090c15);border:1px solid var(--line);border-radius:14px;padding:16px;box-shadow:0 0 0 1px #ffffff05 inset,0 10px 40px #0008;min-width:0;overflow-x:auto}
.s6{grid-column:span 6}.s4{grid-column:span 4}.s8{grid-column:span 8}.s3{grid-column:span 3}@media(max-width:1000px){.s6,.s4,.s8,.s3{grid-column:span 12}}
h2{margin:0 0 10px;font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:var(--m)}h2 em{color:var(--g);font-style:normal}
.kpi{font-size:28px;font-weight:800;color:#fff;text-shadow:0 0 18px #9945ff66}.kpi.g{color:var(--g);text-shadow:0 0 18px #14f19566}.kpi.r{color:var(--r)}
.sub{color:var(--m);font-size:11px}
table{width:100%;border-collapse:collapse;font-size:12px}th{position:sticky;top:0;background:#0b0e17;color:var(--m);text-align:left;font-weight:600;cursor:pointer;padding:6px;border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:6px;border-bottom:1px solid #121726;vertical-align:top}tr:hover td{background:#14f1950a}.pos{color:var(--g)}.neg{color:var(--r)}
.scroll{max-height:520px;overflow:auto;border-radius:10px;border:1px solid var(--line)}
.pill{display:inline-block;padding:1px 8px;border-radius:99px;border:1px solid var(--line);color:var(--c);font-size:11px}
input[type=range]{width:100%;accent-color:var(--g)}select,input[type=text]{background:#070a12;color:var(--t);border:1px solid var(--line);border-radius:8px;padding:6px 8px;font:inherit}
.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.warn{border-color:#ff4d6d55;background:linear-gradient(180deg,#1a0b12,#0d0a10)}
svg text{fill:var(--m);font:10px ui-monospace}.glow{filter:drop-shadow(0 0 6px currentColor)}
.log{max-height:360px;overflow:auto;font-size:11px}.log div{padding:4px 0;border-bottom:1px dashed #1b2133}.qa{color:var(--m)}.yes{color:var(--g)}.no{color:var(--r)}.unk{color:#e5b84a}
__WCSS__</style></head><body>
<header><div class="logo">SOLSTRATS</div><div class="dot" id="livedot" title="live feed"></div><div class="ticker"><span id="tick"></span></div><div class="sub" id="clock"></div></header>
<section id="lstrip"><div class="lsh"><b>LEADERBOARD</b> <span id="lslab">24H</span> <span class="sub">ranked best → worst · walk-forward robust first, then the rest · follows the date-range picker</span></div><div class="lsrow" id="lsrow"></div></section>
<main id="app"></main>
<script>
const REP=__REP__; let SNAP=__SNAP__;
const $=s=>document.querySelector(s), fmtp=x=>x==null?'—':((x>=0?'+':'')+(x*100).toFixed(x*100>=1000||x*100<=-100?0:1)+'%'), fm=x=>x==null?'—':'$'+(Math.abs(x)>=1000?x.toLocaleString(undefined,{maximumFractionDigits:0}):x.toFixed(2));
const cls=x=>x==null?'':(x>=0?'pos':'neg');
function line(el,series,opt={}){const W=el.clientWidth||600,H=opt.h||220,P=28;let all=[].concat(...series.map(s=>s.v.filter(x=>x!=null)));if(!all.length)return;let mn=Math.min(...all),mx=Math.max(...all);if(opt.zero){mn=Math.min(mn,0);mx=Math.max(mx,0)}if(mx==mn)mx=mn+1;
const n=Math.max(...series.map(s=>s.v.length));const X=i=>P+(W-P-8)*i/(n-1||1),Y=v=>H-18-(H-30)*(v-mn)/(mx-mn);
let g=`<svg width="${W}" height="${H}">`;for(let k=0;k<=4;k++){const v=mn+(mx-mn)*k/4;g+=`<line x1="${P}" x2="${W}" y1="${Y(v)}" y2="${Y(v)}" stroke="#1b2133"/><text x="2" y="${Y(v)+3}">${opt.fmt?opt.fmt(v):v.toFixed(0)}</text>`}
const cols=['#14f195','#9945ff','#00d1ff','#ff9f1c','#ff4d6d','#e5e7eb','#7c86a2'];
series.forEach((s,si)=>{const c=s.c||cols[si%cols.length];let d='';s.v.forEach((v,i)=>{if(v!=null)d+=(d?'L':'M')+X(i).toFixed(1)+','+Y(v).toFixed(1)});
if(si==0&&!opt.nofill)g+=`<path d="${d}L${X(s.v.length-1)},${H-18}L${X(0)},${H-18}Z" fill="${c}" opacity=".08"/>`;g+=`<path d="${d}" fill="none" stroke="${c}" stroke-width="${si==0?2.2:1.4}" class="${si==0?'glow':''}" style="color:${c}"/>`});
g+='</svg>';el.innerHTML=g+'<div class="row sub">'+series.map((s,si)=>`<span style="color:${s.c||cols[si%cols.length]}">■</span> ${s.n}`).join(' &nbsp; ')+'</div>'}
function bars(el,items,opt={}){const W=el.clientWidth||600,H=opt.h||200,P=30,n=items.length;let mx=Math.max(...items.map(x=>Math.abs(x.v||0)),1e-9);let g=`<svg width="${W}" height="${H}">`;const bw=(W-P)/n;
items.forEach((it,i)=>{const v=it.v||0,h=(H-40)*Math.abs(v)/mx/(opt.signed?2:1),base=opt.signed?(H-20)/2+10:H-20;const y=v>=0?base-h:base;g+=`<rect x="${P+i*bw+2}" y="${y}" width="${bw-4}" height="${h}" rx="3" fill="${v>=0?'#14f195':'#ff4d6d'}" opacity=".85"/><text x="${P+i*bw+bw/2}" y="${H-4}" text-anchor="middle">${it.l}</text>`});el.innerHTML=g+'</svg>'}
function table(el,rows,cols,opt={}){let st={k:opt.sort||cols[0][0],d:-1},q='';const draw=()=>{let r=rows.filter(x=>!q||JSON.stringify(x).toLowerCase().includes(q));r=r.slice().sort((a,b)=>((a[st.k]??-1e18)>(b[st.k]??-1e18)?1:-1)*st.d);
el.querySelector('tbody').innerHTML=r.slice(0,opt.max||100).map((x,i)=>'<tr>'+cols.map(c=>`<td class="${c[2]?cls(x[c[0]]):''}">${c[0]=='#'?i+1:(c[3]?c[3](x[c[0]],x):(x[c[0]]??'—'))}</td>`).join('')+'</tr>').join('')};
el.innerHTML=(opt.filter?'<div class="row" style="margin-bottom:8px"><input type="text" placeholder="filter… (family, coin, rule)" style="flex:1"></div>':'')+'<div class="scroll"><table><thead><tr>'+cols.map(c=>`<th data-k="${c[0]}">${c[1]}</th>`).join('')+'</tr></thead><tbody></tbody></table></div>';
el.querySelectorAll('th').forEach(th=>th.onclick=()=>{const k=th.dataset.k;if(k=='#')return;st.d=st.k==k?-st.d:-1;st.k=k;draw()});const inp=el.querySelector('input');if(inp)inp.oninput=e=>{q=e.target.value.toLowerCase();draw()};draw()}
const C=(span,title,inner,extra='')=>`<section class="card ${span} ${extra}"><h2>${title}</h2>${inner}</section>`;
const W=REP.winner,B=REP.bench,cov=REP.coverage,cnt=REP.counts;
const LB=[['#','#'],['family','Family'],['entry','Entry rule'],['exit','Exit rule'],['sizing','Sizing'],['balance','Start',0,fm],['oos_end','End (OOS)',0,fm],['oos_profit','Profit $',1,fm],['oos_ret','Profit %',1,fmtp],['oos_win','Win rate',0,fmtp],['oos_trades','Trades'],['oos_dd','Max DD',0,x=>fmtp(-x)],['oos_sharpe','Sharpe',1,x=>x?.toFixed(2)],['is_ret','In-sample %',1,fmtp],['robust','IS & OOS +',0,x=>x?'<span class=pill>both ✓</span>':'—'],['src','Source']];
let html='';
html+=C('','Best strategy (out-of-sample) <em>'+W.family+'</em>',`<div class="row" style="gap:28px"><div><div class="kpi g">${fmtp(W.oos_ret)}</div><div class="sub">OOS return ${cov.split} → ${cov.end} from ${fm(W.balance)}</div></div><div><div class="kpi">${W.oos_sharpe?.toFixed(2)}</div><div class="sub">OOS Sharpe</div></div><div><div class="kpi r">${fmtp(-W.oos_dd)}</div><div class="sub">max drawdown</div></div><div><div class="kpi">${fmtp(W.oos_win)}</div><div class="sub">${W.oos_trades} trades win rate</div></div><div><div class="kpi">${fmtp(B.oos.sol)}</div><div class="sub">SOL buy&hold OOS</div></div><div><div class="kpi">${fmtp(B.oos.basket_est)}</div><div class="sub">meme basket OOS</div></div></div><p>▸ <b>Entry:</b> ${W.entry}<br>▸ <b>Exit:</b> ${W.exit}<br>▸ <b>Size:</b> ${W.sizing}<br><span class="sub">In-sample: ${fmtp(W.is_ret)} return, Sharpe ${W.is_sharpe}, ${W.is_trades} trades.</span></p>`);
__WCARD__
html+=C('s8','Equity of $100 — top strategies vs benchmarks (OOS)','<div id="eq"></div>');
html+=C('s4','Search coverage',`<div class="kpi g">${cnt.grid_configs.toLocaleString()}</div><div class="sub">brute-force configs simulated (in- and out-of-sample)</div><br><div class="kpi">${(REP.overlay?.configs||0).toLocaleString()}</div><div class="sub">agent/check overlay configs</div><br><div class="kpi">${(REP.smart?.evaluated||0).toLocaleString()}</div><div class="sub">Optuna smart-search evaluations</div><br><div class="kpi" style="font-size:20px">${cnt.space_size.toExponential(2)}</div><div class="sub">nominal search-space size</div><br><div class="sub">${cov.tokens} tokens · ${cov.start} → ${cov.end} · ${cov.timeframe} candles · split ${cov.split}</div>`);
html+=C('','Balance explorer <em id="bl"></em>','<input type="range" id="bal" min="0" max="'+(Object.keys(REP.per_balance).length-1)+'" value="19"><div id="balchart"></div><div id="baltable"></div>');
html+=C('','Top 100 strategies — by OOS Sharpe','<div id="t100"></div>');
html+=C('','Top 100 — by OOS profit','<div id="t100p"></div>');
html+=C('','Best of the window (full '+cov.start+' → '+cov.end+', by profit)','<div id="year"></div>');
if(REP.year)html+=C('','Best of the Year — 12 months daily ('+REP.year.start+' → '+REP.year.end+', '+REP.year.coins+' coins)','<div class="sub">'+REP.year.configs.toLocaleString()+' configs on daily closes. Only '+fmtp(REP.year.oos_positive_share_top100)+' of the year\'s top 100 were still profitable after '+REP.year.split+' (hold-out). Basket buy & hold '+fmtp(REP.year.basket_bh)+', SOL '+fmtp(REP.year.sol_bh)+'. Excluded for feed errors: '+REP.year.excluded_feed_errors.join(', ')+'.</div><div id="yr12"></div>');
html+=C('s6','Moonshot: take-profit level vs return','<div id="tp"></div><div class="sub" id="tpnote"></div>');
html+=C('s6','How often launches actually moon',`<div id="moon"></div>`);
html+=C('s6','Winner drawdown (OOS)','<div id="dd"></div>');
html+=C('s6','Winner per-coin (OOS, sum of trade returns)','<div id="pc"></div>');
html+=C('s6','TP × stop heatmap (median OOS return, $100)','<div id="hm" style="overflow-x:auto"></div>');
html+=C('s6','Distribution of OOS returns ($100, all configs)','<div id="hist"></div>');
html+=C('','Agents & checks (median OOS return by family)','<div id="ag"></div><div id="edge"></div>');
html+=C('','90%+ win rate (OOS, ≥30 trades)','<div id="w90"></div>');
html+=C('','Max return (Extreme Upside)','<div id="mr"></div>');
html+=C('s6','Desk pipeline vs the viral post (unverified claim)','<div id="desk"></div>');
html+=C('s6','“Why they died” — launches that fell 90%+ from peak','<div id="dead"></div>');
html+=C('s6','Live fast movers ($2.5k–$3M MC)','<div id="movers"></div>');
html+=C('s6','Live paper books — backtest vs live','<div id="books"></div>');
if(REP.micro_be)html+=C('','Micro buy sizes — break-even per strategy <em>fixed $ per trade, real fees</em>','<div class="sub">Fixed cost per round trip ≈ '+fm(REP.micro_be.fixed_rt_usd)+' (2 swaps × base+priority fee + 25% of token-account rent, SOL avg $'+REP.micro_be.sol_avg.toFixed(0)+'). Break-even = smallest fixed buy whose simulated total P&L is positive.</div><div id="micro"></div>');
if(REP.stress)html+=C('','Robustness checks — robust leaderboard under stress <em>full span, own balance & sizing</em>','<div class="sub" id="stressnote"></div><div id="stress"></div><h3 style="margin:14px 0 6px">Month by month <em>(May and Nov are partial; America/Chicago months)</em></h3><div id="monthly"></div>');
if(REP.bt_live)html+=C('','Backtest vs Live <em>paper books @ $100</em>','<div class="sub" id="btlnote"></div><div id="btl"></div>');
html+=C('','Logger — agent self-check Q&A','<div class="log" id="log"></div>');
html+=C('','Families',`<div id="fam"></div>`);
html+=C('','Not financial advice',`Backtest and paper trading only. Meme coins routinely go to zero; past results do not predict future results. With ${cnt.grid_configs.toLocaleString()} configurations tested, the best result is expected to look good by luck alone (null expected max Sharpe ≈ ${REP.multiple_testing.expected_max_sharpe_null}).`,'warn');
$('#app').innerHTML=html;
const cv=REP.curves;line($('#eq'),Object.keys(cv).map(k=>({n:k,v:cv[k]})),{fmt:v=>'$'+v.toFixed(0),h:260});
const balKeys=Object.keys(REP.per_balance);function drawBal(){const k=balKeys[$('#bal').value];$('#bl').textContent='$'+k;table($('#baltable'),REP.per_balance[k],LB,{sort:'oos_sharpe',max:25})}
$('#bal').oninput=drawBal;drawBal();line($('#balchart'),[{n:'winner OOS return by start balance',v:REP.vs_balance.map(x=>x.winner_oos_ret*100)},{n:'best OOS return at that balance',v:REP.vs_balance.map(x=>x.best_oos_ret*100)}],{fmt:v=>v.toFixed(0)+'%',h:180,zero:1});
table($('#t100'),REP.top100,LB,{sort:'oos_sharpe',filter:1});table($('#t100p'),REP.top100_profit,LB,{sort:'oos_ret',filter:1});
table($('#year'),REP.best_year,[['#','#'],['family','Family'],['entry','Entry'],['exit','Exit'],['balance','Start',0,fm],['full_end','End',0,fm],['full_ret','Return',1,fmtp],['full_win','Win',0,fmtp],['full_trades','Trades'],['full_dd','Max DD',0,x=>fmtp(-x)],['oos_ret','OOS part',1,fmtp]],{sort:'full_ret',filter:1});
if(REP.year)table($('#yr12'),REP.year.top,[['#','#'],['family','Family'],['entry','Entry'],['exit','Exit'],['sizing','Sizing'],['balance','Start',0,fm],['end','End',0,fm],['ret','12m return',1,fmtp],['win','Win',0,fmtp],['trades','Trades'],['dd','Max DD',0,x=>fmtp(-x)],['is_ret','First 7m',1,fmtp],['oos_ret','Hold-out 5m',1,fmtp]],{sort:'ret',filter:1});
const tp=REP.moon_tp;line($('#tp'),[{n:'best OOS return %',v:tp.map(x=>x.best_oos_ret*100)},{n:'best full-window return %',v:tp.map(x=>x.best_full_ret*100)},{n:'TP hit rate %',v:tp.map(x=>(x.hit_rate||0)*100)}],{fmt:v=>v.toFixed(0),h:220,zero:1});
$('#tpnote').textContent='TP levels: '+tp.map(x=>x.tp_pct+'%').join(' · ');
const L=REP.launch_stats;bars($('#moon'),[{l:'2x',v:L.hit_2x},{l:'5x',v:L.hit_5x},{l:'10x',v:L.hit_10x},{l:'100x',v:L.hit_100x},{l:'died',v:L.died_share}]);$('#moon').insertAdjacentHTML('beforeend',`<div class="sub">${L.n} launches with data: ${fmtp(L.hit_10x)} ever printed 10x above first-hour price; ${fmtp(L.died_share)} fell 90%+ from peak.</div>`);
line($('#dd'),[{n:'drawdown %',v:REP.winner_curve.dd.map(x=>x*100),c:'#ff4d6d'}],{fmt:v=>v.toFixed(0)+'%',zero:1});
bars($('#pc'),REP.per_coin.slice(0,16).map(x=>({l:x.symbol.slice(0,6),v:x.sum_ret})),{signed:1});
(function(){const h=REP.heat;let s='<table><tr><th></th>'+h.cols.map(c=>`<th>${c}</th>`).join('')+'</tr>';h.values.forEach((r,i)=>{s+=`<tr><td class="sub">${h.rows[i]}</td>`+r.map(v=>{if(v==null)return '<td>—</td>';const a=Math.min(1,Math.abs(v)*2);return `<td style="background:${v>=0?`rgba(20,241,149,${a*.6})`:`rgba(255,77,109,${a*.6})`}">${fmtp(v)}</td>`}).join('')+'</tr>'});$('#hm').innerHTML=s+'</table>'})();
bars($('#hist'),REP.hist.map(x=>({l:x.label.split(' ')[0],v:x.count})));
if(REP.overlay){const o=REP.overlay;let s='<div class="scroll"><table><tr><th>agents on</th>'+o.families.map(f=>`<th>${f}</th>`).join('')+'</tr>';o.agents_table.forEach(r=>{s+=`<tr><td>${r.agents}</td>`+r.values.map(v=>v==null?'<td>—</td>':`<td style="background:${v>=0?'rgba(20,241,149,'+Math.min(.6,Math.abs(v))+')':'rgba(255,77,109,'+Math.min(.6,Math.abs(v))+')'}">${fmtp(v)}</td>`).join('')+'</tr>'});$('#ag').innerHTML=s+'</table></div>';
table($('#edge'),o.edge,[['check','Check'],['median_with','Median OOS with',1,fmtp],['median_without','without',1,fmtp],['edge','Edge',1,fmtp]],{sort:'edge'})}
const w90=REP.win90.length?REP.win90:REP.win90_closest;$('#w90').insertAdjacentHTML('beforebegin',`<div class="sub">${REP.win90.length?REP.win90.length+' strategies reached ≥90% OOS win rate with ≥30 trades.':'None reached 90% OOS with ≥30 trades — closest shown.'}</div>`);
table($('#w90'),w90,[['family','Family'],['entry','Entry'],['exit','Exit'],['oos_win','Win',0,fmtp],['oos_trades','Trades'],['oos_ret','Profit %',1,fmtp],['oos_avg_win','Avg win $',0,fm],['oos_avg_loss','Avg loss $',0,fm],['oos_dd','Max DD',0,x=>fmtp(-x)]],{sort:'oos_ret'});
const X=REP.extreme;$('#mr').innerHTML=`<div class="row" style="gap:30px"><div><div class="kpi g">${fmtp(X.oos.ret)}</div><div class="sub">highest OOS return</div></div><div><div class="kpi">${fmtp(X.is.ret)}</div><div class="sub">highest in-sample (likely overfit)</div></div><div><div class="kpi">${fmtp(X.full.ret)}</div><div class="sub">highest full window</div></div><div><div class="kpi ${X.reached_100000pct?'g':'r'}">${X.reached_100000pct?'YES':'NO'}</div><div class="sub">any config ≥ +100,000%?</div></div></div><div id="mrt"></div>`;
table($('#mrt'),REP.max_ret.oos,LB,{sort:'oos_ret'});
const DX=REP.desk_exact;if(DX){$('#desk').innerHTML=`<table><tr><th></th><th>Post (claimed, unverified)</th><th>Our backtest OOS</th><th>Our full window</th></tr><tr><td>Start</td><td>$280</td><td>$280</td><td>$280</td></tr><tr><td>End</td><td>$3,412 (overnight)</td><td>${fm(DX.oos.end)}</td><td>${fm(DX.full.end)}</td></tr><tr><td>Trades</td><td>38</td><td>${DX.oos.trades}</td><td>${DX.full.trades}</td></tr><tr><td>W / L</td><td>26 / 12</td><td>${DX.oos.wins} / ${DX.oos.losses}</td><td>${DX.full.wins} / ${DX.full.losses}</td></tr><tr><td>Best / worst</td><td>+$810 / −$37</td><td>${fm(DX.oos.best)} / ${fm(DX.oos.worst)}</td><td>${fm(DX.full.best)} / ${fm(DX.full.worst)}</td></tr></table><div id="deskc"></div>`;line($('#deskc'),[{n:'our Desk rules, $280, full window',v:DX.full.curve},{n:'claimed path (5 points, spread over window)',v:DX.full.curve.map((_,i,a)=>{const p=[280,388,1395,2942,3412];const k=Math.round(i/(a.length-1)*4);return Math.abs(i/(a.length-1)*4-k)<.02?p[k]:null}),c:'#ff9f1c'}],{fmt:v=>'$'+v.toFixed(0)})}
table($('#dead'),REP.dead,[['symbol','Token'],['peak_x','Peak ×',0,x=>x?.toFixed(1)+'×'],['end_vs_peak','From peak',1,fmtp],['hours_to_peak','h to peak'],['vol_last24_vs_peak24','Vol now/peak',0,fmtp],['top10_ex_lp','Top-10',0,fmtp]],{sort:'peak_x'});
table($('#fam'),REP.families,[['family','Family'],['configs','Configs'],['with_trades','With trades'],['median_oos_ret','Median OOS',1,fmtp],['pct_profitable_oos','% profitable OOS',0,fmtp],['best','Best OOS (Sharpe pick)',0,b=>b?fmtp(b.oos_ret)+' · '+b.entry:'—']],{sort:'median_oos_ret'});
function drawLive(){const s=SNAP||{};const mv=s.fast_movers||[];$('#movers').innerHTML=mv.length?'':'<div class="sub">start run_live.py and serve this folder to stream</div>';if(mv.length)table($('#movers'),mv,[['symbol','Token'],['mcap','MC',0,fm],['chg_h1','1h %',1,x=>x==null?'—':x.toFixed(1)+'%'],['vol_h1','1h vol',0,fm],['liq','Liq',0,fm],['age_h','Age h',0,x=>x?.toFixed(1)]],{sort:'chg_h1'});
const bk=s.books||{};const rows=Object.entries(bk).map(([k,b])=>({book:k,equity:b.equity,ret:b.equity/b.balance0-1,trades:(b.closed||[]).length,wins:(b.closed||[]).filter(c=>c.pnl>0).length,open:(b.open||[]).length}));rows.forEach(r=>r.win=r.trades?r.wins/r.trades:null);
if(rows.length)table($('#books'),rows,[['book','Strategy@balance'],['equity','Equity',0,fm],['ret','Live %',1,fmtp],['win','Win',0,fmtp],['trades','Closed'],['open','Open']],{sort:'ret'});
$('#log').innerHTML=(s.log||[]).slice().reverse().map(e=>`<div><b class="${e.action=='BUY'?'yes':'no'}">${e.agent} ${e.action}</b> ${e.symbol||''} @ ${e.price} · ${e.reason} · ${e.book}<div class="qa">${(e.qa||[]).map(q=>`${q.agent}: ${q.q} <span class="${q.a=='yes'?'yes':q.a=='no'?'no':'unk'}">${q.a}</span>${q.value!=null?' ('+q.value+')':''}`).join(' · ')}</div></div>`).join('')||'<div class="sub">no paper trades yet</div>';
$('#tick').innerHTML=(mv.concat(mv)).map(t=>`${t.symbol} <b>${fm(t.mcap)}</b> ${(t.chg_h1||0)>=0?'<b>':'<i>'}${(t.chg_h1||0).toFixed(1)}%${(t.chg_h1||0)>=0?'</b>':'</i>'}`).join(' · ')||'live feed idle';$('#clock').textContent=s.ts?'feed '+new Date(s.ts*1000).toLocaleTimeString():'static snapshot'}
__WJS__
if(REP.micro_be){const sz=[0.01,0.05,0.1,0.25,0.5,1];table($('#micro'),REP.micro_be.rows.map(r=>{const o={...r};sz.forEach(z=>{const x=r.sims.find(q=>q.size==z);o['s'+z]=x?x.pnl_per_trade:null});return o}),[['id','ID'],['tag','Board'],['family','Family'],['trades','Trades'],['avg_net_pct','Avg net/trade',1,fmtp],['breakeven_usd','Break-even buy',0,x=>x==null?'never ≤$250':fm(x)]].concat(sz.map(z=>['s'+z,'P&L/trade @$'+z,1,x=>x==null?'—':(x>=0?'+':'-')+'$'+Math.abs(x).toFixed(3)])),{sort:'breakeven_usd'})}
if(REP.stress){const S=REP.stress.summary,P=REP.stress.summary.profitable_at,nl={};(REP.nullt?REP.nullt.rows:[]).forEach(r=>nl[r.id]=r);const pl=REP.plateau?REP.plateau.centres.find(c=>c.id=='H6-01098'):null;
$('#stressnote').innerHTML=`${S.survivors} of ${S.n} robust strategies stay profitable at 2× costs, one bar late and without whichever token helps them most. Profitable at 1×/1.5×/2×/3× costs: ${P.base}/${P['cost_x1.5']}/${P.cost_x2}/${P.cost_x3}; one bar late: ${P.late_1bar}. `+(pl?`Plateau: ${(pl.profitable*100).toFixed(0)}% of ${pl.neighbours} neighbouring H6-01098 configs are profitable, median ${fmtp(pl.median_full_ret)} vs the picked point's +557% — expect the picked number to be inflated by selection. `:'')+(REP.nullt?`Random-entry null (${REP.nullt.draws} draws, same exits/sizing/signal count per token): p is the share of random entries that do as well (floor ${(1/(REP.nullt.draws+1)).toFixed(4)}). It does not correct for how many rules were searched.`:'');
table($('#stress'),REP.stress.rows.map(r=>({...r,p:nl[r.id]?nl[r.id].p_value:null,nullmed:nl[r.id]?nl[r.id].null_median:null})),[['id','ID'],['balance','Bal',0,fm],['trades','Trades'],['base','Full span',1,fmtp],['cost_x2','2× costs',1,fmtp],['cost_x3','3× costs',1,fmtp],['late_1bar','1 bar late',1,fmtp],['drop_best_trade','w/o best trade',1,fmtp],['drop_best_token','w/o best token',1,fmtp],['best_token','Best token'],['sub80_p05','80% subsample p5',1,fmtp],['p','Null p',0,x=>x==null?'—':x.toFixed(4)],['nullmed','Null median',1,x=>x==null?'—':fmtp(x)],['survives','Survives',0,x=>x?'<b class=yes>yes</b>':'<i class=no>no</i>']],{sort:'base'})}
if(REP.monthly){const M=REP.monthly;table($('#monthly'),M.rows.map(r=>{const o={id:r.id,total:r.total,up:r.up_months+'/'+r.months.length};M.labels.forEach((l,i)=>o[l]=r.months[i]);return o}),[['id','ID']].concat(M.labels.map(l=>[l,l,1,fmtp])).concat([['total','Full span',1,fmtp],['up','Up months']]),{sort:'total'})}
if(REP.bt_live){const m=REP.bt_live.meta;$('#btlnote').textContent=`Live paper run ${new Date(m.started*1000).toLocaleTimeString('en-US',{timeZone:'America/Chicago',hour:'numeric',minute:'2-digit'})} → ${new Date(m.ts*1000).toLocaleTimeString('en-US',{timeZone:'America/Chicago',hour:'numeric',minute:'2-digit'})} CT: ${m.polls} polls, ${m.errors} errors, ${m.http.requests} HTTP requests, ${m.http['429']} rate-limit retries (all recovered). Live minutes are not comparable to backtest days; open marks on bonding-curve tokens are not fillable at the mark.`;
table($('#btl'),REP.bt_live.rows,[['book','Book'],['family','Family'],['live_equity','Live equity',0,fm],['live_ret','Live % (mid mark)',1,fmtp],['live_liq_ret','Live % if sold now',1,fmtp],['live_closed','Closed'],['live_open','Open'],['bt_24h_ret','Backtest last 24h',1,fmtp],['bt_24h_trades','BT 24h trades'],['bt_12m_ret','Backtest full span',1,fmtp],['bt_12m_trades','BT trades']],{sort:'live_ret'})}
(()=>{const cs=[...document.querySelectorAll('section.card')];const n=document.createElement('nav');n.id='jump';cs.forEach((c,i)=>{const h=c.querySelector('h2');if(!h)return;c.id=c.id||('c'+i);const a=document.createElement('a');a.href='#'+c.id;a.textContent=h.childNodes[0].textContent.split(/ [—(]/)[0].trim().slice(0,34);n.appendChild(a)});$('#app').before(n)})();
drawLive();setInterval(()=>fetch('data/live/snapshot.json',{cache:'no-store'}).then(r=>r.json()).then(j=>{SNAP=j;drawLive()}).catch(()=>{}),15000);
</script></body></html>"""
import window_js as WJ
wp = json.load(open(f"{R}/window_payload.json")) if os.path.exists(f"{R}/window_payload.json") else None
ws = json.load(open(f"{R}/windows.json")) if os.path.exists(f"{R}/windows.json") else None
H = HTML
if wp and ws:
    H = H.replace("__WCARD__", WJ.CARD).replace("__WCSS__", WJ.CSS).replace("__WJS__", WJ.JS.replace("__WP__", json.dumps(wp)).replace("__WS__", json.dumps(ws)))
else:
    H = H.replace("__WCARD__", "").replace("__WCSS__", "").replace("__WJS__", "")
out = H.replace("__REP__", json.dumps(rep)).replace("__SNAP__", json.dumps(snap))
open("dashboard.html", "w").write(out)
print("dashboard.html", len(out) // 1024, "KB")
