"""Inject/refresh the 'LIVE PAPER — $20' panel in a local dashboard HTML file (idempotent, between markers)."""
import html, json, re, sys
from pathlib import Path
import os
HOME = Path(os.environ.get("SOLSTRATS_HOME") or Path(__file__).resolve().parent)
ROOT = Path(os.environ.get("SOLSTRATS_RUN_DIR") or HOME / "run")
DASH = Path(os.environ.get("SOLSTRATS_DASHBOARD") or ROOT / "dashboard.html")   # local file by default
A, B = "<!--LIVEPAPER-->", "<!--/LIVEPAPER-->"
def esc(x): return html.escape(str(x))
def money(v): return f"${v:,.2f}"
def fmt(ts):
    from datetime import datetime; from zoneinfo import ZoneInfo
    return datetime.fromtimestamp(ts, ZoneInfo("America/Chicago")).strftime("%Y-%m-%d %I:%M %p %Z")
def build(state):
    res = state.get("results") or {}
    cards, rows, opn = [], [], []
    for s in state["strategies"]:
        r = res.get(s["id"], {}); eq = r.get("equity", s["balance"]); rt = r.get("ret_pct", 0.0); bt = s.get("backtest") or s.get("backtest_at_20") or s.get("backtest_note")
        col = "#39FF14" if eq >= s["balance"] else "#FF2D95"
        cards.append(f'<div class="lpc"><div class="lpr">{esc(s.get("label") or ("#" + str(s["rank"]) + " " + s["id"]))}</div><div class="lpe" style="color:{col}">{money(eq)}</div>'
                     f'<div class="lpm">{rt:+.2f}% · {len(r.get("open", []))} open · {r.get("n_closed", 0)} closed</div>'
                     f'<div class="lpx"><b>Buy:</b> {esc(s["entry"])}<br><b>Sell:</b> {esc(s["exit"])}<br><b>Size:</b> {esc(s["sizing"])}, {money(s["balance"])} virtual start · started {esc(fmt(s.get("start_ts") or state["start_ts"]))}</div>'
                     + (f'<div class="lpx">Backtest @ {money(s["balance"])}: {bt["ret"]*100:+,.0f}% full · {bt["r30flat"]*100:+.0f}% last 30d · {bt["hflat"]*100:+.0f}% last 14d · {bt["wf_pct"]:.0f}% of {bt["wf_n"]} folds profitable · max DD {bt["dd"]*100:.0f}%</div>' if bt else '<div class="lpx">Backtest: see daily routine note</div>') +
                     f'<div class="lpx">Pending buys next bar: {esc(", ".join(r.get("pending", [])) or "none")}</div></div>')
        for t in r.get("open", []):
            opn.append(f'<tr><td>{esc(s["id"])}</td><td>{esc(t["sym"])}</td><td>{esc(t["entry_time"])}</td><td>{t["entry_px"]:.6g}</td><td>{t["exit_px"]:.6g}</td><td>{money(t["size_usd"])}</td><td>{t["pnl"]:+.2f}</td></tr>')
        for t in r.get("closed", []):
            rows.append((t["exit_time"], f'<tr><td>{esc(s["id"])}</td><td>{esc(t["sym"])}</td><td>{esc(t["entry_time"])}</td><td>{esc(t["exit_time"])}</td><td>{money(t["size_usd"])}</td><td>{t["pnl"]:+.2f}</td><td>{esc(t["reason"])}</td></tr>'))
    rows.sort(key=lambda x: x[0], reverse=True)
    from datetime import datetime; from zoneinfo import ZoneInfo
    start = datetime.fromtimestamp(state["start_ts"], ZoneInfo("America/Chicago")).strftime("%Y-%m-%d %I:%M %p %Z")
    css = ("<style>#livepaper{margin:12px;padding:12px 14px;border:1px solid #FFB000;border-radius:14px;background:#0a0d16;box-shadow:0 0 18px #ffb00033;font-size:12px;color:#d7def0}"
           "#livepaper h2{margin:0 0 4px;color:#FFB000;letter-spacing:.1em;font-size:15px;text-shadow:0 0 10px #ffb00088}#livepaper .lpb{color:#9fb3c8;margin-bottom:8px}"
           ".lpg{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:10px}.lpc{border:1px solid #2a3350;border-radius:12px;padding:10px;background:linear-gradient(180deg,#0d1322,#070a12)}"
           ".lpr{color:#00E5FF;font-weight:700}.lpe{font-size:22px;font-weight:700}.lpm{color:#9fb3c8}.lpx{color:#8a96b0;font-size:11px;margin-top:6px}"
           "#livepaper table{width:100%;border-collapse:collapse;margin-top:6px}#livepaper td,#livepaper th{border-bottom:1px solid #1c2338;padding:3px 6px;text-align:left}</style>")
    body = (f'<section id="livepaper"><h2>LIVE PAPER — $20 (+ H6-01098 books) &nbsp;·&nbsp; PAPER TRADING, NO REAL MONEY</h2>'
            f'<div class="lpb">Started {esc(start)} · last update {esc(state.get("last_cycle", "pending first cycle"))} · candles through {esc(state.get("data_end", "?"))} · '
            f'hourly GeckoTerminal candles, exact backtest rules and cost model, each book has its own virtual balance. Fills are booked at the next hourly open after a signal, as in the backtest.</div>'
            f'<div class="lpg">{"".join(cards)}</div>'
            f'<h3>Open positions (marked at last hourly close, exit costs included)</h3><table><tr><th>Strategy</th><th>Token</th><th>Entry</th><th>Entry px</th><th>Mark px</th><th>Size</th><th>P&amp;L $</th></tr>{"".join(opn) or "<tr><td colspan=7>none yet</td></tr>"}</table>'
            f'<h3>Trade log</h3><table><tr><th>Strategy</th><th>Token</th><th>Entry</th><th>Exit</th><th>Size</th><th>P&amp;L $</th><th>Why</th></tr>{"".join(r for _, r in rows[:200]) or "<tr><td colspan=7>no closed trades yet</td></tr>"}</table></section>')
    return A + css + body + B
def inject():
    state = json.loads((ROOT / "state.json").read_text())
    block = build(state)
    h = DASH.read_text() if DASH.exists() else '<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="300"><title>SOLSTRATS live paper</title></head><body style="background:#05070d;margin:0;font-family:ui-monospace,Menlo,Consolas,monospace"></body></html>'
    if A in h: h = re.sub(re.escape(A) + ".*?" + re.escape(B), lambda m: block, h, flags=re.S)
    elif "</section>" in h and 'id="lstrip"' in h:
        i = h.index("</section>", h.index('id="lstrip"')) + len("</section>"); h = h[:i] + block + h[i:]
    elif "<body" in h:
        i = h.index(">", h.index("<body")) + 1; h = h[:i] + block + h[i:]
    else: h = block + h
    tmp = DASH.with_suffix(".html.tmp"); tmp.write_text(h); tmp.replace(DASH)
if __name__ == "__main__": inject(); print("injected")
