// Headless Chrome render check via CDP (node >= 22): loads dashboard.html, collects uncaught JS errors, checks key cards rendered.
import { spawn } from "node:child_process"; import { mkdtempSync } from "node:fs"; import { tmpdir } from "node:os"; import path from "node:path";
const file = path.resolve(process.argv[2] || "dashboard.html"), port = 9333 + Math.floor(Math.random() * 500), width = +(process.argv[3] || 1400);
const ch = spawn("google-chrome", ["--headless=new", "--no-sandbox", "--disable-gpu", `--remote-debugging-port=${port}`, `--user-data-dir=${mkdtempSync(path.join(tmpdir(), "rc-"))}`, `--window-size=${width},2000`, "about:blank"], { stdio: "ignore" });
const sleep = ms => new Promise(r => setTimeout(r, ms));
let tgt; for (let i = 0; i < 60 && !tgt; i++) { await sleep(500); try { tgt = (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()).find(t => t.type === "page"); } catch {} }
const ws = new WebSocket(tgt.webSocketDebuggerUrl); await new Promise(r => ws.onopen = r);
let id = 0; const pend = {}, errs = [];
ws.onmessage = m => { const d = JSON.parse(m.data); if (d.id && pend[d.id]) { pend[d.id](d.result); delete pend[d.id]; }
  if (d.method === "Runtime.exceptionThrown") errs.push(d.params.exceptionDetails.exception?.description || d.params.exceptionDetails.text); };
const send = (method, params = {}) => new Promise(r => { pend[++id] = r; ws.send(JSON.stringify({ id, method, params })); });
await send("Runtime.enable"); await send("Page.enable"); await send("Page.navigate", { url: "file://" + file }); await sleep(12000);
const expr = `JSON.stringify({window_board:document.querySelectorAll('#wtab tr').length, micro:document.querySelectorAll('#micro tr').length,
  btl:document.querySelectorAll('#btl tr').length, stress:document.querySelectorAll('#stress tr').length, monthly:document.querySelectorAll('#monthly tr').length, stressnote:(document.querySelector('#stressnote')||{}).textContent?.length||0,
  wlab:(document.querySelector('#wlab')||{}).textContent||'', overflowX: document.documentElement.scrollWidth>window.innerWidth+2, cards:document.querySelectorAll('section.card').length, nav:document.querySelectorAll('#jump a').length, strip:document.querySelectorAll('#lsrow .lsc').length, strip1:(document.querySelector('#lsrow .lsc')||{}).textContent})`;
const r = JSON.parse((await send("Runtime.evaluate", { expression: expr, returnByValue: true })).result.value);
// exercise the picker: click every preset button and confirm the board re-renders
const pr = (await send("Runtime.evaluate", { expression: `(async()=>{const out={};for(const b of document.querySelectorAll('#wpre button')){b.click();await new Promise(r=>setTimeout(r,400));out[b.textContent]=document.querySelectorAll('#wtab tr').length+' / '+(document.querySelector('#lsrow .lsc .nm')||{}).textContent+' '+(document.querySelector('#lsrow .lsc .pf')||{}).textContent}return JSON.stringify(out)})()`, awaitPromise: true, returnByValue: true })).result.value;
const ov = (await send("Runtime.evaluate", { expression: `JSON.stringify([...document.querySelectorAll('body *')].filter(e=>{if(e.getBoundingClientRect().right<=window.innerWidth+2)return false;for(let p=e.parentElement;p;p=p.parentElement){const o=getComputedStyle(p).overflowX;if(o!='visible')return false}return true}).slice(0,8).map(e=>e.tagName+'.'+e.className+'#'+e.id+' in '+(e.closest('[id]')||{}).id+' / '+(e.closest('section')?.querySelector('h2')?.textContent||'').slice(0,40)+' '+Math.round(e.getBoundingClientRect().right)))`, returnByValue: true })).result.value;
console.log(JSON.stringify({ width, overflowing: JSON.parse(ov), scrollW: (await send('Runtime.evaluate',{expression:'document.documentElement.scrollWidth+"/"+window.innerWidth',returnByValue:true})).result.value, errors: errs, ...r, presets: JSON.parse(pr || "{}") }));
if (process.env.SHOT) { await send("Runtime.evaluate", { expression: `(document.querySelector(${JSON.stringify(process.env.SHOT_SEL||'#stress')})?.closest('section')||document.body).scrollIntoView()` }); await sleep(600);
  const img = await send("Page.captureScreenshot", { format: "png" }); (await import("node:fs")).writeFileSync(process.env.SHOT, Buffer.from(img.data, "base64")); }
ws.close(); ch.kill("SIGKILL"); process.exit(errs.length || !r.window_board || !r.stress ? 1 : 0);
