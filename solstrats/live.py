"""Live PAPER trading runner (no real trades, no wallet, no keys in output).

Every `interval` seconds:
  SCOUT    discovers new pairs (GeckoTerminal new_pools, DexScreener latest profiles, Birdeye if keyed)
  prices   for the watchlist + discovered tokens (DexScreener; Birdeye fallback if keyed)
  FILTER   on-chain checks via Helius RPC (public RPC fallback): mint/freeze authority, top-10 holders
  ANALYST  score + momentum from live pair stats
  EXECUTOR paper-fills entries at live price + modeled costs for each tracked strategy x starting balance
  RISK     stop / trailing / take-profit / time / mcap-target exits on each poll
  LOGGER   writes every decision with its self-check Q&A to data/live/log.jsonl
Outputs data/live/snapshot.json for the dashboards."""
from __future__ import annotations
import json, logging, math, os, time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import numpy as np
from .http import Client
from .providers import Providers
from . import costs as K
from .engine import NONE, F
from .describe import entry_text, exit_text

log = logging.getLogger("solstrats.live")

def _clip01(x):
    return 0.0 if x < 0 else (1.0 if x > 1 else x)

def analyst_score(t: dict, top10: Optional[float]) -> float:
    """Live analogue of engine.desk_score using DexScreener pair stats."""
    s1 = _clip01(((t.get("chg_h6") or t.get("chg_h1") or 0) / 100 + 0.2) / 0.7)
    vol_h1 = t.get("vol_h1") or 0.0; vol_h24 = t.get("vol_h24") or 0.0
    vr = vol_h1 / (vol_h24 / 24) if vol_h24 > 0 else 1.0
    s2 = _clip01((vr - 1) / 4)
    age_h = t.get("age_h", 1e9)
    s3 = _clip01(1 - age_h / 720)
    s4 = _clip01(1 - (top10 if top10 is not None else 0.6) / 0.6)
    s5 = _clip01((math.log10(vol_h1 + 1) - 3) / 2.5)
    return (s1 + s2 + s3 + s4 + s5) / 5

def self_checks(t: dict, safety: dict, cfg: dict) -> List[dict]:
    """Explicit yes/no/probability questions each Desk agent asks before acting, answered from live data."""
    q = []
    A = lambda agent, question, ans, val=None, gate=True: q.append({"agent": agent, "q": question, "a": ans, "value": val, "gate": gate})
    age = t.get("age_h"); liq = t.get("liq") or 0; v1 = t.get("vol_h1") or 0
    b, s = t.get("buys_h1") or 0, t.get("sells_h1") or 0
    A("SCOUT", "New pair (< %dh old)?" % cfg.get("new_pair_h", 720), "yes" if age is not None and age <= cfg.get("new_pair_h", 720) else "no", round(age, 1) if age is not None else None)
    A("SCOUT", "Enough liquidity (>= $%s)?" % int(cfg.get("liq_min", 10000)), "yes" if liq >= cfg.get("liq_min", 10000) else "no", round(liq))
    wash = (v1 / liq) if liq > 0 else None
    A("SCOUT", "Real volume, not wash (1h vol / liquidity < 20)?", "yes" if wash is not None and wash < 20 else ("unknown" if wash is None else "no"), round(wash, 2) if wash is not None else None)
    ma = safety.get("mint_auth"); fa = safety.get("freeze_auth")
    A("FILTER", "Mint authority revoked?", "unknown" if ma is None else ("yes" if not ma else "no"))
    A("FILTER", "Freeze authority revoked?", "unknown" if fa is None else ("yes" if not fa else "no"))
    A("FILTER", "LP burned/locked?", "unknown", None, False)   # not available from free endpoints; never gated
    t10 = safety.get("top10_ex_lp")
    A("FILTER", "Top-10 holders < %d%% (ex-LP)?" % int(cfg.get("top10_max", 0.2) * 100), "unknown" if t10 is None else ("yes" if t10 < cfg.get("top10_max", 0.2) else "no"), round(t10, 3) if t10 is not None else None)
    A("FILTER", "Dev wallet clean (no past rugs)?", "unknown", None, False)  # needs wallet history; not gated
    A("FILTER", "Volume > $%s/hr?" % int(cfg.get("vol_min", 50000)), "yes" if v1 >= cfg.get("vol_min", 50000) else "no", round(v1))
    A("ANALYST", "Momentum rising (1h change > 0)?", "yes" if (t.get("chg_h1") or 0) > 0 else "no", t.get("chg_h1"))
    tot = b + s
    A("ANALYST", "Buyers > sellers (1h)?", "yes" if b > s else "no", round(b / tot, 3) if tot else None)
    A("ANALYST", "Holder growth?", "unknown", None, False)
    A("ANALYST", "Smart wallets in?", "unknown", None, False)
    sc = analyst_score(t, t10)
    A("ANALYST", "Score >= %.2f?" % cfg.get("score_min", 0.6), "yes" if sc >= cfg.get("score_min", 0.6) else "no", round(sc, 3))
    return q

@dataclass
class Position:
    mint: str; symbol: str; entry: float; size: float; t_entry: float; hw: float; active: bool = False
    venue: int = 1; vol_entry: float = 0.0

def venue_of(t: dict, watch_mints) -> int:
    """Same venue classes as the backtest: 2 = pump.fun bonding curve, 0 = established (watch list), 1 = launch pool."""
    if (t.get("dex") or "") in ("pumpfun", "pump-fun"): return 2
    return 0 if t.get("mint") in watch_mints else 1

def exit_value(pos: Position, px: float, solpx: float) -> float:
    """USD received when selling now, with the backtest's costs: DEX fee + slippage per side for the venue, price impact
    IMPACT_K * size / entry-hour volume on both sides, and the fixed SOL fee per swap."""
    pc = float(K.PCT_COST[pos.venue]) + (K.IMPACT_K * pos.size / (pos.vol_entry + 1.0) if pos.vol_entry > 0 else 0.0)
    fix = K.FIXED_SOL * solpx
    return (pos.size - fix) * (1 - pc) * (px / pos.entry) * (1 - pc) - fix

REENTRY_S = 3600.0

@dataclass
class Book:
    strat_id: str; balance0: float; cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    closed: List[dict] = field(default_factory=list)
    curve: List[list] = field(default_factory=list)
    last_exit: Dict[str, float] = field(default_factory=dict)   # mint -> exit time; no re-entry within REENTRY_S (backtest: next fill >= exit bar + 2)

    def equity(self, prices: Dict[str, float]) -> float:
        v = self.cash
        for m, p in self.positions.items():
            px = prices.get(m)
            v += p.size * (px / p.entry if px else 1.0)
        return v

    def liq_equity(self, prices: Dict[str, float], solpx: float) -> float:
        """Equity if every open position were sold now (exit costs, impact, rent refund) - comparable to the backtest's unrealized marks."""
        v = self.cash
        for m, p in self.positions.items():
            px = prices.get(m) or p.entry
            v += exit_value(p, px, solpx) + K.RENT_SOL * solpx * (1 - K.RENT_UNRECOVERED)
        return v

def live_signal(st: dict, t: dict, checks: List[dict]) -> bool:
    """Translate a backtested launch-family rule to live pair stats (hourly granularity preserved)."""
    fam = st["family"]; p = st["p"]
    age = t.get("age_h", 1e9); mc = t.get("mcap") or 0; v1 = t.get("vol_h1") or 0; c1 = (t.get("chg_h1") or 0) / 100
    if fam == "moonshot":
        ok = age <= p[0] and mc <= p[1] and v1 >= p[2] and c1 >= p[3]
    elif fam == "fast_mc":
        ok = p[0] <= mc <= p[1] and c1 >= p[2] / max(1, p[3])   # growth per hour (pro-rated) from live 1h change
    elif fam == "graduation":
        ok = age <= p[0] and v1 >= p[2] and c1 >= p[1]
    elif fam == "desk":
        ok = age <= p[3]
    elif fam == "bonding_curve":
        ok = (t.get("dex") in ("pumpfun", "pump-fun")) and mc >= p[0] and v1 >= p[1]
    else:
        ok = t["mint"] in st.get("_hourly_hits", set())  # history families: engine.signals on live hourly bars
    mask = st.get("ovmask", 0)
    names = {0: ("SCOUT", "New pair"), 1: ("FILTER", "Mint authority"), 2: ("FILTER", "Freeze authority"), 3: ("FILTER", "Top-10"),
             4: ("FILTER", "Volume"), 5: ("ANALYST", "Momentum"), 6: ("ANALYST", "Score")}
    for b, (ag, key) in names.items():
        if mask & (1 << b):
            hit = [c for c in checks if c["agent"] == ag and c["q"].startswith(key)]
            if not hit or hit[0]["a"] != "yes":
                return False
    return ok

class LiveRunner:
    def __init__(self, strategies: List[dict], balances: List[float], out_dir: str = "data/live", watch: Optional[List[dict]] = None,
                 providers: Optional[Providers] = None, cfg: Optional[dict] = None):
        self.pv = providers or Providers(Client(cache_dir=os.path.join(out_dir, "cache")))
        self.strats = strategies; self.balances = balances; self.out = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.watch = watch or []
        self.cfg = {"new_pair_h": 720, "liq_min": 10000, "top10_max": 0.2, "vol_min": 50000, "score_min": 0.6, **(cfg or {})}
        self.books: Dict[str, Book] = {}
        for st in strategies:
            for b in balances:
                k = f"{st['id']}@{b:g}"
                self.books[k] = Book(st["id"], b, b)
        self.safety: Dict[str, dict] = {}
        self.t_start = time.time(); self.polls = 0; self.errors = 0
        self.logf = os.path.join(out_dir, "log.jsonl")
        self.recent_log: List[dict] = []
        self.xdex: List[dict] = []

    def _log(self, rec: dict):
        rec["ts"] = time.time()
        self.recent_log.append(rec); self.recent_log = self.recent_log[-300:]
        try:
            with open(self.logf, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass

    def _safety(self, mint: str) -> dict:
        if mint not in self.safety:
            try:
                self.safety[mint] = self.pv.mint_safety(mint)
            except Exception as e:  # noqa: BLE001
                self.safety[mint] = {"error": type(e).__name__}
        return self.safety[mint]

    HISTORY_FAMS = ("breakout", "ma_cross", "rsi_revert", "vol_spike", "dip_buy", "narrative", "flash_wick")

    def refresh_hourly(self):
        """Once per hour: pull ~200 hourly candles per watched pool (GeckoTerminal, cached) and evaluate the
        exact backtest signal function on the last COMPLETED bar for every history-based strategy."""
        from . import engine as E
        hs = [s for s in self.strats if s["family"] in self.HISTORY_FAMS]
        if not hs or not self.watch: return
        hour = int(time.time() // 3600)
        if getattr(self, "_hour", None) == hour: return
        self._hour = hour
        bars = {}
        for w in self.watch:
            if not w.get("pool"): continue
            try:
                rows = self.pv.ohlcv_hour(w["pool"], 200)
                if len(rows) >= 30: bars[w["mint"]] = np.asarray(rows, float)
            except Exception as e:  # noqa: BLE001
                self.errors += 1; log.warning("ohlcv %s failed: %s", w.get("symbol"), type(e).__name__)
        if not bars: return
        mints = list(bars); tmax = max(r[-1, 0] for r in bars.values()); T = 200
        O = np.full((T, len(mints)), np.nan); H = O.copy(); L = O.copy(); C = O.copy(); V = np.zeros_like(O)
        for j, m in enumerate(mints):
            r = bars[m]; idx = (T - 1 - (tmax - r[:, 0]) // 3600).astype(int); ok = idx >= 0
            O[idx[ok], j], H[idx[ok], j], L[idx[ok], j], C[idx[ok], j], V[idx[ok], j] = r[ok, 1], r[ok, 2], r[ok, 3], r[ok, 4], r[ok, 5]
            for t in range(1, T):
                if np.isnan(C[t, j]) and not np.isnan(C[t - 1, j]):
                    C[t, j] = O[t, j] = H[t, j] = L[t, j] = C[t - 1, j]
        valid = ~np.isnan(C); first = np.where(valid.any(0), valid.argmax(0), -1).astype(np.int64)
        n = len(mints); z = np.zeros(n); zi = np.zeros(n, np.int64)
        for st in hs:
            sig = E.signals(E.F[st["family"]], np.asarray(st["p"], float), O, H, L, C, V, first, zi, zi, np.full(n, 1e9), z, z,
                            np.full(n, np.nan), np.zeros(T, np.int64), zi, T - 3, T - 1)
            st["_hourly_hits"] = {mints[j] for j in range(n) if sig[T - 2, j]}   # last completed hour
            self._log({"agent": "SCOUT", "action": "SCAN", "book": st["id"], "symbol": ",".join(w["symbol"] for w in self.watch if w["mint"] in st["_hourly_hits"]) or "-",
                       "price": None, "reason": f"hourly {st['family']} scan of {n} watched pools: {len(st['_hourly_hits'])} signal(s)", "qa": []})

    def poll(self) -> dict:
        self.polls += 1
        now = time.time()
        # hourly candle refresh runs in a background thread so a slow/throttled GeckoTerminal never blocks the poll loop
        import threading
        th = getattr(self, "_th", None)
        if th is None or not th.is_alive():
            def _job():
                try:
                    self.refresh_hourly()
                except Exception as e:  # noqa: BLE001
                    self.errors += 1; log.warning("hourly refresh failed: %s", type(e).__name__)
            self._th = threading.Thread(target=_job, daemon=True); self._th.start()
        try:
            newp = self.pv.new_pairs(limit=60)
        except Exception as e:  # noqa: BLE001
            newp = []; self.errors += 1; log.warning("new_pairs failed: %s", type(e).__name__)
        mints = [w["mint"] for w in self.watch] + [n["mint"] for n in newp if n.get("mint")]
        held = {m for b in self.books.values() for m in b.positions}
        mints = list(dict.fromkeys(list(held) + mints))[:150]
        try:
            toks = self.pv.tokens(mints)
        except Exception as e:  # noqa: BLE001
            toks = []; self.errors += 1
        by = {t["mint"]: t for t in toks}
        for n in newp:  # keep geckoterminal-only pairs too
            if n.get("mint") and n["mint"] not in by and n.get("price"):
                by[n["mint"]] = n
        for t in by.values():
            cm = t.get("created_ms")
            if cm: t["age_h"] = (now * 1000 - cm) / 3.6e6
            elif t.get("created"):
                try:
                    t["age_h"] = (now - time.mktime(time.strptime(t["created"][:19], "%Y-%m-%dT%H:%M:%S")) + time.timezone) / 3600
                except (ValueError, TypeError):
                    pass
        prices = {m: t["price"] for m, t in by.items() if t.get("price")}
        if self.polls % 5 == 1 and self.watch:
            try:
                self.xdex = scan_cross_dex(self.pv, [w["mint"] for w in self.watch])
            except Exception:  # noqa: BLE001
                self.errors += 1
        solpx = 80.0
        sp = self.pv.prices(["So11111111111111111111111111111111111111112"])
        solpx = sp.get("So11111111111111111111111111111111111111112", solpx)
        # RISK: manage open positions
        for key, bk in self.books.items():
            st = next(s for s in self.strats if s["id"] == bk.strat_id)
            tp, sl, act, w, tm, mct, cr = st["e"]
            for m, pos in list(bk.positions.items()):
                px = prices.get(m)
                if not px: continue
                pos.hw = max(pos.hw, px)
                if not pos.active and pos.hw >= pos.entry * (1 + act): pos.active = True
                stop = pos.entry * (1 - sl) if sl < 1 else 0
                if pos.active: stop = max(stop, pos.hw * (1 - w))
                mc_now = (by.get(m) or {}).get("mcap") or 0
                reason = None
                if px <= stop: reason = "trailing-stop" if pos.active and stop > pos.entry * (1 - sl) else "stop-loss"
                elif tp < NONE / 2 and px >= pos.entry * (1 + tp): reason = "take-profit"
                elif mct < NONE / 2 and mc_now >= mct: reason = "mcap-target"
                elif (now - pos.t_entry) / 3600 >= tm: reason = "time-stop"
                if reason:
                    val = exit_value(pos, px, solpx)
                    bk.cash += val + K.RENT_SOL * solpx * (1 - K.RENT_UNRECOVERED)
                    pnl = val - pos.size - K.RENT_SOL * solpx * K.RENT_UNRECOVERED
                    bk.closed.append({"mint": m, "symbol": pos.symbol, "pnl": pnl, "ret": px / pos.entry - 1, "reason": reason, "t": now})
                    del bk.positions[m]; bk.last_exit[m] = now
                    self._log({"agent": "RISK", "action": "SELL", "book": key, "symbol": pos.symbol, "mint": m, "price": px, "reason": reason,
                               "pnl": round(pnl, 4), "qa": [{"agent": "RISK", "q": f"Down {sl*100:g}%?", "a": "yes" if px <= pos.entry * (1 - sl) else "no"},
                                                           {"agent": "RISK", "q": f"Up {act*100:g}% (trail active)?", "a": "yes" if pos.active else "no"},
                                                           {"agent": "RISK", "q": "Time stop hit?", "a": "yes" if reason == "time-stop" else "no"},
                                                           {"agent": "RISK", "q": "LP pulled / dev selling?", "a": "unknown"}]})
        # SCOUT -> FILTER -> ANALYST -> EXECUTOR for candidates
        hits = set().union(*[s_.get("_hourly_hits", set()) for s_ in self.strats]) if self.strats else set()
        cands = [t for t in by.values() if t.get("price") and ((t.get("age_h") is not None and t["age_h"] <= 720) or t["mint"] in hits)]
        decisions = 0
        for t in cands[:40]:
            safety = self._safety(t["mint"]) if (t.get("liq") or 0) >= 5000 else {}
            qa = self_checks(t, safety, self.cfg)
            for key, bk in self.books.items():
                st = next(s for s in self.strats if s["id"] == bk.strat_id)
                if t["mint"] in bk.positions: continue
                if now - bk.last_exit.get(t["mint"], -1e18) < REENTRY_S: continue
                if st["family"] in self.HISTORY_FAMS and st.get("_entered_hour", {}).get(t["mint"]) == getattr(self, "_hour", None):
                    continue
                if not live_signal(st, t, qa):
                    continue
                st.setdefault("_entered_hour", {})[t["mint"]] = getattr(self, "_hour", None)
                frac, maxpos = st["frac"], st["maxpos"]
                eq = bk.equity(prices); size = frac * eq
                ex_q = [{"agent": "EXECUTOR", "q": f"Size within {frac*100:g}% of stack?", "a": "yes", "value": round(size, 4)},
                        {"agent": "EXECUTOR", "q": "Already at max positions?", "a": "yes" if len(bk.positions) >= maxpos else "no"},
                        {"agent": "EXECUTOR", "q": "Slippage acceptable (size < 5% of 1h volume)?", "a": "yes" if size < 0.05 * (t.get("vol_h1") or 0) else "no"}]
                rent = K.RENT_SOL * solpx
                if len(bk.positions) >= maxpos or size + rent > bk.cash or size >= 0.05 * (t.get("vol_h1") or 0) or size <= 2 * K.FIXED_SOL * solpx:
                    continue
                bk.cash -= size + rent
                bk.positions[t["mint"]] = Position(t["mint"], t.get("symbol") or "?", t["price"], size, now, t["price"],
                                                  venue=venue_of(t, {w_["mint"] for w_ in self.watch}), vol_entry=float(t.get("vol_h1") or 0))
                decisions += 1
                self._log({"agent": "EXECUTOR", "action": "BUY", "book": key, "symbol": t.get("symbol"), "mint": t["mint"], "price": t["price"],
                           "size": round(size, 4), "reason": f"{st['family']} signal passed", "qa": qa + ex_q})
        for bk in self.books.values():
            bk.curve.append([now, round(bk.equity(prices), 4)])
            bk.curve = bk.curve[-2000:]
        movers = sorted([t for t in by.values() if 2500 <= (t.get("mcap") or 0) <= 3e6],
                        key=lambda t: -((t.get("chg_h1") or 0)))[:25]
        snap = {"ts": now, "started": self.t_start, "polls": self.polls, "sol_price": solpx, "provider_status": self.pv.status(),
                "http_stats": dict(self.pv.c.stats), "new_pairs": [self._slim(t) for t in list(by.values())[:40] if t.get("age_h", 1e9) <= 72],
                "fast_movers": [self._slim(t) for t in movers], "cross_dex": self.xdex[:20],
                "books": {k: {"strat": b.strat_id, "balance0": b.balance0, "equity": round(b.equity(prices), 4), "equity_liq": round(b.liq_equity(prices, solpx), 4), "cash": round(b.cash, 4),
                              "open": [{"symbol": p.symbol, "entry": p.entry, "size": round(p.size, 4), "now": prices.get(m), "venue": p.venue,
                                        "exit_value": round(exit_value(p, prices.get(m) or p.entry, solpx), 4)} for m, p in b.positions.items()],
                              "closed": b.closed[-50:], "curve": b.curve[-500:]} for k, b in self.books.items()},
                "log": self.recent_log[-100:], "errors": self.errors}
        tmp = os.path.join(self.out, "snapshot.json.tmp")
        with open(tmp, "w") as f:
            json.dump(snap, f)
        os.replace(tmp, os.path.join(self.out, "snapshot.json"))
        return snap


    @staticmethod
    def _slim(t):
        return {k: t.get(k) for k in ("symbol", "mint", "dex", "price", "mcap", "liq", "vol_h1", "chg_m5", "chg_h1", "buys_h1", "sells_h1", "age_h", "source")}

    def run(self, duration_s: float, interval: float = 20.0):
        end = time.time() + duration_s
        while time.time() < end:
            t = time.time()
            try:
                self.poll()
            except Exception as e:  # noqa: BLE001 - keep running; record
                self.errors += 1
                log.exception("poll failed: %s", type(e).__name__)
            time.sleep(max(0.0, interval - (time.time() - t)))

def scan_cross_dex(pv: Providers, mints: List[str], liq_min: float = 20000, mev_take: float = 0.9) -> List[dict]:
    """Market-inefficiency scanner: same token priced differently across pools. Paper only.
    Net gap = gap - 2 x (fee+slippage) - priority fees; expected capture haircut by MEV competition (mev_take)."""
    out = []
    for m0 in mints:
        try:
            d = pv.c.get("dexscreener_fast", f"https://api.dexscreener.com/token-pairs/v1/solana/{m0}", cache_ttl=60)
        except Exception:  # noqa: BLE001
            continue
        pools: Dict[str, list] = {}
        for p in d if isinstance(d, list) else []:
            try:
                if p.get("chainId") != "solana" or float((p.get("liquidity") or {}).get("usd") or 0) < liq_min: continue
                if p["quoteToken"]["symbol"] not in ("SOL", "USDC", "USDT"): continue
                pools.setdefault(p["baseToken"]["address"], []).append((float(p["priceUsd"]), p["dexId"], p["pairAddress"], p["baseToken"]["symbol"]))
            except (KeyError, TypeError, ValueError):
                continue
        for m, ps in pools.items():
            if len(ps) < 2: continue
            lo = min(ps); hi = max(ps)
            gap = hi[0] / lo[0] - 1
            cost = 2 * 0.0025 + 2 * 0.003 + 0.001   # two pool fees + slippage + priority/latency allowance
            net = gap - cost
            out.append({"symbol": lo[3], "mint": m, "buy_on": lo[1], "sell_on": hi[1], "gap_pct": round(gap * 100, 3),
                        "net_pct": round(net * 100, 3), "expected_pct_after_mev": round(max(net, 0) * (1 - mev_take) * 100, 4)})
    return sorted(out, key=lambda r: -r["net_pct"])
