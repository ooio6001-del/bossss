"""Local dashboard + window API (paper/backtest only; no keys are used or exposed here).

  python3 serve.py [port]            -> http://localhost:8765/dashboard.html
  GET /api/window?preset=24h                      (1h 4h 12h 24h 48h 7d 30d 90d ytd 12m)
  GET /api/window?start=2026-11-10T08:00&end=2026-11-12T20:00   (hour precision, America/Chicago if no offset; end inclusive)
      optional: &sort=ret|profit|win|dd  &tag=robust|top100|top100_profit|hunter  &limit=100  &ids=H6-01098,G4097-S1
Runs the engine on the requested window: signals before the window are cleared (positions start flat), same cost model,
open positions marked to the window's last close (unrealized P&L)."""
from __future__ import annotations
import json, os, sys, threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".numba_cache"))
sys.dont_write_bytecode = True
from strategies import Registry
from compute_windows import strategy_rows
from solstrats import window as W
from run_backtest import prep

R = os.environ.get("SOLSTRATS_RESULTS", "results")

class Engine:
    def __init__(self):
        self.lock = threading.Lock(); self.D = prep(); self.reg = Registry(self.D, R); self.specs = []
        for r in strategy_rows(R):
            s = self.reg.spec(r)
            if s: s["tag"] = r["_tag"]; s["entry"] = r.get("entry", ""); s["exit"] = r.get("exit", ""); self.specs.append(s)

    def window(self, q):
        g = lambda k, d=None: (q.get(k) or [d])[0]
        b = W.resolve(self.D, preset=g("preset"), start=g("start"), end=g("end"))
        ids = set(filter(None, (g("ids") or "").split(","))); tag = g("tag")
        rows = []
        with self.lock:
            for s in self.specs:
                if (ids and s["id"] not in ids) or (tag and s["tag"] != tag): continue
                tr = W.replay(self.reg.candidates(s), b["t_lo"], s["one_shot"])
                m = W.pack(W.run(self.D, tr, b["t_lo"], b["t_hi"], s["bal"], s["frac"], s["maxpos"]), s["bal"])
                rows.append({"id": s["id"], "tag": s["tag"], "family": s["family"], "entry": s["entry"], "exit": s["exit"],
                             "sizing": s["sizing"], **m})
        key = {"ret": lambda r: r["ret"], "profit": lambda r: r["profit"], "win": lambda r: (r["winrate"], r["ntrades"]),
               "dd": lambda r: -r["maxdd"]}.get(g("sort", "ret"), lambda r: r["ret"])
        rows.sort(key=key, reverse=True)
        for i, r in enumerate(rows): r["rank"] = i + 1
        return {"window": b, "n": len(rows), "rows": rows[: int(g("limit", "100"))]}

ENGINE = None

class H(SimpleHTTPRequestHandler):
    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/api/window":
            try:
                body = json.dumps(ENGINE.window(parse_qs(u.query))).encode(); code = 200
            except (ValueError, KeyError) as e:
                body = json.dumps({"error": str(e)}).encode(); code = 400
            self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
        if u.path.startswith("/.env") or u.path.endswith((".env", ".key")):
            self.send_error(404); return
        return super().do_GET()
    def log_message(self, *a): pass

def main():
    global ENGINE
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    ENGINE = Engine()
    print(f"serving on http://127.0.0.1:{port}/dashboard.html  ({len(ENGINE.specs)} strategies, data {W.resolve(ENGINE.D, '12m')['data_start']} -> {W.resolve(ENGINE.D, '12m')['data_end']})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()

if __name__ == "__main__":
    main()
