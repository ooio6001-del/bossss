"""Strategy registry for window recomputes: rebuilds a signal + exit + sizing from a leaderboard row.
Grid rows (G<combo>-S<size>) come from results/sweep_meta.json; hunter rows (H*) are rebuilt read-only with
/workspace/solstrats_hunt/wf_lib.py (that folder is never written: bytecode and numba caches are redirected)."""
from __future__ import annotations
import csv, json, os, re, sys
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".numba_cache"))
sys.dont_write_bytecode = True
import numpy as np
from solstrats import engine as E, grid as G
from solstrats.describe import sizing_text

HUNT = "/workspace/solstrats_hunt"
_wf = None
def wf():
    global _wf
    if _wf is None:
        sys.path.insert(0, HUNT)
        import wf_lib  # noqa
        _wf = wf_lib
    return _wf

def read_csv(p):
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []

class Registry:
    def __init__(self, D, results="results"):
        self.D = D
        self.meta = json.load(open(f"{results}/sweep_meta.json"))
        self.sig_cache = {}; self.caches = {}
        self.hunt = {}
        for r in read_csv(f"{HUNT}/leaderboard.csv") + read_csv(f"{HUNT}/robust_leaderboard.csv"):
            self.hunt.setdefault((r["id"], float(r["balance"]), r["sizing"]), r)
            self.hunt.setdefault(r["id"], r)

    def spec(self, row):
        """-> dict(key, sig_key, family, e, one_shot, frac, maxpos, bal) or None"""
        rid = row["id"]; bal = float(row["balance"])
        m = re.match(r"G(\d+)-S(\d+)$", rid)
        if m:
            c = self.meta["combos"][int(m.group(1))]; fr, mp = G.SIZING[int(m.group(2))]
            return dict(id=rid, sig_key=("G", int(m.group(1))), family=c["family"], p=c["p"], e=c["e"], one_shot=c["one_shot"],
                        frac=fr, maxpos=mp, bal=bal, sizing=sizing_text(fr, mp))
        if rid.startswith("H"):
            h = self.hunt.get((rid, bal, row.get("sizing", ""))) or self.hunt.get(rid)
            if not h or not h.get("params") or not h.get("exit_params"):
                return None
            fr, mp = wf().parse_sizing(row.get("sizing") or h["sizing"])
            return dict(id=rid, sig_key=("H", str(h.get("batch")), h["family"], h["params"]), family=h["family"], hrow=h,
                        e=json.loads(h["exit_params"]), one_shot=False, frac=fr, maxpos=mp, bal=bal, sizing=row.get("sizing") or h["sizing"])
        return None

    def signal(self, s):
        k = s["sig_key"]
        if k not in self.sig_cache:
            D = self.D
            if k[0] == "G":
                fam = s["family"]; p = np.asarray(s["p"], float)
                hot = D["hot"].get(int(p[0]), D["hot"][24]) if fam == "narrative" else D["hot"][24]
                self.sig_cache[k] = E.signals(E.F[fam], p, D["O"], D["H"], D["L"], D["C"], D["V"], D["first"], D["grp"], D["venue"],
                                              D["supply"], D["mint_auth"], D["freeze"], D["top10"], hot, D["theme"], 0, D["T"] - 1)
            else:
                self.sig_cache[k] = np.asarray(wf().signal_from_row(s["hrow"], D, self.caches), dtype=np.bool_)
        return self.sig_cache[k]

    def trades(self, s, t_lo=0):
        """Trades with signals before t_lo cleared (start flat). t_lo=0 -> full-period trade list."""
        D = self.D; sig = self.signal(s)
        if t_lo > 1:
            sig = sig.copy(); sig[: t_lo - 1] = False
        return E.gen_trades(sig, np.asarray(s["e"], float), D["O"], D["H"], D["L"], D["C"], D["last"], D["supply"], bool(s["one_shot"]))

    def candidates(self, s):
        """gen_trades_all candidates (sorted by token, fill bar); replay with window.replay for any window start."""
        k = (s["sig_key"], tuple(map(float, s["e"])), bool(s["one_shot"]))
        if not hasattr(self, "_cand"): self._cand = {}
        if k not in self._cand:
            D = self.D
            self._cand[k] = E.gen_trades_all(self.signal(s), np.asarray(s["e"], float), D["O"], D["H"], D["L"], D["C"], D["last"],
                                             D["supply"], bool(s["one_shot"]))
        return self._cand[k]
