"""Client-side date-range picker for dashboard.html (inserted by build_html.py)."""
CARD = r"""html+=C('','Date range <em>window recompute</em> <span class="pill" id="wlab"></span>',`<div class="row" id="wpre"></div>
<div class="row" style="margin-top:10px"><label class="sub">start <input type="datetime-local" step="3600" id="ws"></label><label class="sub">end (incl.) <input type="datetime-local" step="3600" id="we"></label>
<button class="btn" id="wgo">recompute</button><select id="wsort"><option value="ret">rank: return %</option><option value="profit">rank: profit $</option><option value="win">rank: win rate</option><option value="dd">rank: lowest drawdown</option></select>
<select id="wtag"><option value="">all strategies</option><option value="robust">walk-forward robust</option><option value="top100">Top 100 (Sharpe)</option><option value="top100_profit">Top 100 (profit)</option><option value="hunter">hunter top 200</option></select></div>
<div class="row" style="margin:10px 0;gap:26px" id="wkpi"></div><div class="sub" id="wnote"></div><div id="wtab"></div>`);"""
JS = r"""
const WP=__WP__, WS=__WS__;let WD=null;
const b64u=async(s,T)=>{const bin=Uint8Array.from(atob(s),c=>c.charCodeAt(0));const buf=await new Response(new Blob([bin]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();return new T(buf)};
async function wload(){if(WD)return WD;const [tok,ti,to,gross,pin,ve,eo,lcq]=await Promise.all([b64u(WP.tok,Uint8Array),b64u(WP.ti,Uint16Array),b64u(WP.to,Uint16Array),b64u(WP.gross,Float32Array),b64u(WP.pin,Float32Array),b64u(WP.ve,Float32Array),b64u(WP.eo,Uint8Array),b64u(WP.lcq,Int16Array)]);WD={tok,ti,to,gross,pin,ve,eo,lcq};return WD}
const mark=(j,t)=>{const q=WD.lcq[j*WP.T+t];return q==32767?NaN:Math.exp(WP.lcbase[j]+q/1000)};
// port of solstrats.window.simulate_window (same cost model as engine.simulate)
// candidates = every signal bar's trade (engine.gen_trades_all); replay the no-overlap rule from the window start
// => identical to regenerating the strategy with all pre-window signals cleared (positions start flat)
function wsim(si,lo,hi){const S=WS.strategies[si],g=WP.group[si],a=WP.off[g],b=WP.off[g+1],one=WP.one_shot[si],bal=S.balance,frac=S.frac,maxpos=S.maxpos,P=WP;const L=[];
let cur=-1,nxt=0,done=false;
for(let q=a;q<b;q++){const j=WD.tok[q];if(j!=cur){cur=j;nxt=lo;done=false}if(done||WD.ti[q]<nxt)continue;nxt=WD.to[q]+2;if(one)done=true;const t=WD.ti[q];if(t>=hi)continue;const open=WD.to[q]>=hi||(WD.eo[q]&&WD.to[q]==hi-1);let g=WD.gross[q],x=WD.to[q];if(open){const m=mark(WD.tok[q],hi-1);g=isFinite(m)&&m>0?m/WD.pin[q]:1;x=hi-1}L.push({t,j:WD.tok[q],x,g,open,ve:WD.ve[q]})}
L.sort((u,v)=>(u.t+u.j*1e-7)-(v.t+v.j*1e-7));let cash=bal,peak=bal,mdd=0,ntr=0,wins=0,nun=0,unr=0,fees=0,skip=0;const op=[];
const closeBefore=tn=>{for(;;){let mi=-1,me=Infinity;op.forEach((o,i)=>{if(o.x<tn&&o.x<me){me=o.x;mi=i}});if(mi<0)break;const o=op.splice(mi,1)[0];cash+=o.val+o.lock*(1-P.rent_unrec);const pnl=o.val-o.cost-o.lock*P.rent_unrec;if(o.open){nun++;unr+=pnl}else{ntr++;if(pnl>0)wins++}
let eq=cash;op.forEach(z=>eq+=z.cost+z.lock);if(eq>peak)peak=eq;const dd=peak>0?1-eq/peak:0;if(dd>mdd)mdd=dd}};
for(const tr of L){closeBefore(tr.t);if(op.length>=maxpos){skip++;continue}let eq=cash;op.forEach(z=>eq+=z.cost+z.lock);const fixed=frac<0;let size=fixed?-frac:frac*eq;const sp=P.solpx[tr.t],fix=P.fixed_sol*sp,lock=P.rent_sol*sp;
if(size+lock>cash){if(fixed){skip++;continue}size=cash-lock}if(size<=(fixed?0:fix*2)){skip++;continue}if(size>P.fill_cap*tr.ve){skip++;continue}
const pc=P.pct_cost[P.venue[tr.j]]+P.impact_k*size/(tr.ve+1);const val=(size-fix)*(1-pc)*tr.g*(1-pc)-fix;fees+=size*pc*(1+tr.g)+2*fix+P.rent_unrec*lock;cash-=size+lock;op.push({x:tr.x,val,cost:size,lock,open:tr.open})}
closeBefore(Infinity);return {final:cash,ret:cash/bal-1,profit:cash-bal,maxdd:mdd,ntrades:ntr,wins,winrate:ntr?wins/ntr:0,n_open:nun,unrealized:unr,fees,skipped:skip}}
const hourOf=ts=>Math.floor((ts-WP.t0)/3600),tsOf=h=>WP.t0+3600*h;
const toLocal=ts=>{const d=new Date(ts*1000);const p=n=>String(n).padStart(2,'0');return `${d.getFullYear()}-${p(d.getMonth()+1)}-${p(d.getDate())}T${p(d.getHours())}:00`};
const PRE=[['1h',1],['4h',4],['12h',12],['24h',24],['48h',48],['7d',168],['30d',720],['90d',2160],['ytd','ytd'],['12m','12m']];
let WCUR={preset:'24h'};
function wstrip(rows,label){const sk=$('#wsort').value;const key={ret:r=>r.ret,profit:r=>r.profit,win:r=>r.ntrades?r.winrate+r.ntrades*1e-6:-1,dd:r=>r.ntrades+r.n_open?-r.maxdd:-9}[sk];
const R=rows.map((m,i)=>({...WS.strategies[i],...m})).sort((a,b)=>((b.tag=='robust')-(a.tag=='robust'))||(key(b)-key(a))),seen=new Set(),U=R.filter(r=>!seen.has(r.id+'@'+r.balance)&&seen.add(r.id+'@'+r.balance)).slice(0,60);
$('#lslab').textContent=label+' · by '+$('#wsort').selectedOptions[0].text.replace('rank: ','');
$('#lsrow').innerHTML=U.map((r,i)=>{const rob=r.tag=='robust',u=r.profit>=0;return `<div class="lsc ${i==0?'r1':''} ${rob?'rob':''}" title="${(r.entry||'').replace(/"/g,'')} · ${(r.exit||'').replace(/"/g,'')}"><span class="bd ${rob?'y':'n'}">${rob?'ROBUST ✓':r.tag=='top100'?'TOP100':r.tag=='top100_profit'?'TOP$':'HUNTER'}</span><div class="rk">#${i+1}</div><div class="nm">${r.id}</div><div class="fm">${r.family} · ${fm(r.balance)} start</div><div class="pf ${u?'up':'dn'}">${u?'+':'-'}$${Math.abs(r.profit).toFixed(2)}</div><div class="mt"><span class="${u?'up':'dn'}">${fmtp(r.ret)}</span><span>win ${r.ntrades?(r.winrate*100).toFixed(0)+'%':'—'}</span><span>${r.ntrades+r.n_open} tr</span></div></div>`}).join('')+`<div class="lsc" style="flex-basis:150px;display:flex;align-items:center;color:#7d8aa5">${rows.length-U.length} more in the board below</div>`}
function wrender(rows,lo,hi,label,exact){wstrip(rows,label);const tag=$('#wtag').value,sk=$('#wsort').value;let R=rows.map((m,i)=>({...WS.strategies[i],...m,i})).filter(r=>!tag||r.tag==tag);
const key={ret:r=>r.ret,profit:r=>r.profit,win:r=>r.ntrades?r.winrate+r.ntrades*1e-6:-1,dd:r=>r.ntrades+r.n_open?-r.maxdd:-9}[sk];R.sort((a,b)=>key(b)-key(a));
const act=R.filter(r=>r.ntrades+r.n_open>0),pos=act.filter(r=>r.profit>0);const best=R[0];
$('#wlab').textContent=label+(exact?' · exact (engine, regenerated flat)':' · client recompute');
$('#wkpi').innerHTML=`<div><div class="kpi g">${fmtp(best?.ret)}</div><div class="sub">#1 ${best?.id} ${best?.family} @ ${fm(best?.balance)}</div></div><div><div class="kpi">${hi-lo}h</div><div class="sub">${new Date(tsOf(lo)*1000).toLocaleString()} → ${new Date(tsOf(hi-1)*1000).toLocaleString()}</div></div><div><div class="kpi">${act.length}/${R.length}</div><div class="sub">strategies with a position in window</div></div><div><div class="kpi ${pos.length>=act.length/2?'g':'r'}">${act.length?(100*pos.length/act.length).toFixed(0):0}%</div><div class="sub">of active ones profitable</div></div>`;
table($('#wtab'),R.slice(0,100).map(r=>({...r,rules:r.entry+' · '+r.exit+' · '+r.sizing})),[['#','#'],['id','ID'],['tag','Board'],['family','Family'],['rules','Rules'],['balance','Start',0,fm],['final','End',0,fm],['profit','Profit $',1,fm],['ret','Return',1,fmtp],['ntrades','Closed'],['winrate','Win',0,fmtp],['maxdd','Max DD',0,x=>fmtp(-x)],['n_open','Open'],['unrealized','Unrealized $',1,fm]],{sort:{ret:'ret',profit:'profit',win:'winrate',dd:'maxdd'}[sk],filter:1});}
async function wrun(){await wload();let lo,hi,exact=null,label;
if(WCUR.preset){const P=WS.presets[WCUR.preset];lo=P.bounds.t_lo;hi=P.bounds.t_hi;exact=P.rows;label=WCUR.preset.toUpperCase()+(P.bounds.capped_to_data?' (capped to available hourly data from '+P.bounds.data_start+')':'')}
else{lo=Math.max(0,hourOf(new Date($('#ws').value).getTime()/1000));hi=Math.min(WP.T,hourOf(new Date($('#we').value).getTime()/1000)+1);if(!(hi>lo)){$('#wnote').textContent='end must be after start';return}label='custom';
if(new Date($('#ws').value).getTime()/1000<WP.t0||hi>=WP.T&&hourOf(new Date($('#we').value).getTime()/1000)>=WP.T)label+=' (capped to data)'}
$('#ws').value=toLocal(tsOf(lo));$('#we').value=toLocal(tsOf(hi-1));
const rows=exact||WS.strategies.map((_,i)=>wsim(i,lo,hi));
$('#wnote').textContent=(exact?'Preset: engine results computed on the box. ':'Custom: recomputed in your browser with a port of the same engine and cost model. ')+'Signals before the window are cleared, so every strategy starts flat at the window start. '+'Open positions at the window end are marked to that hour\'s close, exit costs included, and shown as unrealized. Times shown in your browser\'s local zone. Hourly data: '+WS.presets['12m'].bounds.data_start+' → '+WS.presets['12m'].bounds.data_end+'.';
wrender(rows,lo,hi,label,!!exact)}
$('#wpre').innerHTML=PRE.map(p=>`<button class="btn" data-p="${p[0]}">${p[0].toUpperCase()}</button>`).join('')+'<span class="sub">YTD/12m are capped to the ~179 days of hourly data</span>';
$('#wpre').querySelectorAll('button').forEach(b=>b.onclick=()=>{WCUR={preset:b.dataset.p};$('#wpre').querySelectorAll('button').forEach(x=>x.classList.toggle('on',x==b));wrun()});
$('#wgo').onclick=()=>{WCUR={};$('#wpre').querySelectorAll('button').forEach(x=>x.classList.remove('on'));wrun()};
$('#wsort').onchange=$('#wtag').onchange=()=>wrun();
$('#wpre').querySelector('[data-p="24h"]').classList.add('on');wrun().catch(e=>{$('#wnote').textContent='window engine failed to load: '+e});
"""
CSS = ".btn{background:#0b1220;color:var(--t);border:1px solid var(--line);border-radius:8px;padding:6px 12px;font:inherit;cursor:pointer;letter-spacing:.06em}.btn:hover{border-color:var(--g);box-shadow:0 0 12px #14f19555}.btn.on{border-color:var(--g);color:var(--g);box-shadow:0 0 14px #14f19566 inset}input[type=datetime-local]{background:#070a12;color:var(--t);border:1px solid var(--line);border-radius:8px;padding:6px 8px;font:inherit;color-scheme:dark}"
