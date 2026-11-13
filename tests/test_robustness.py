"""Stress, null-test and server guards (synthetic data, no network)."""
import json, threading, urllib.request, urllib.error
import numpy as np
import stress, null_test, plateau, serve
from tests.test_window import rand_data, E1, gen

def _D(seed=5):
    D, sig = rand_data(seed=seed); D["venue"] = np.zeros(sig.shape[1], np.int64); return D, sig

def test_cost_multiplier_monotone_and_x1_is_base():
    D, sig = _D(); tr = gen(D, sig)
    base = stress.sim(D, tr, 100.0, 0.2, 5)[0]
    assert stress.sim(D, tr, 100.0, 0.2, 5, 1.0)[0] == base
    r = [stress.sim(D, tr, 100.0, 0.2, 5, m)[0] for m in (1.0, 1.5, 2.0, 3.0)]
    assert all(a >= b - 1e-12 for a, b in zip(r, r[1:]))

def test_sub_keeps_alignment():
    D, sig = _D(); tr = gen(D, sig); keep = np.arange(tr[0].size) % 2 == 0
    s = stress.sub(tr, keep)
    assert all(a.size == keep.sum() for a in s) and np.array_equal(s[1], tr[1][keep])

class _Reg:
    def __init__(self, D, sig): self.D, self.sig = D, sig
    def signal(self, s): return self.sig
    def trades(self, s, t_lo=0): return gen(self.D, self.sig)

def test_null_p_value_bounds():
    D, sig = _D(7); D["first"] = np.zeros(sig.shape[1], np.int64)
    null_test.DRAWS = 20
    s = {"id": "x", "family": "f", "e": E1, "one_shot": False, "bal": 100.0, "frac": 0.2, "maxpos": 5}
    r = null_test.null_one(D, _Reg(D, sig), s, np.random.default_rng(0))
    assert 1 / 21 <= r["p_value"] <= 1 and 0 <= r["null_profitable"] <= 1

def test_plateau_key_parse():
    assert plateau.key(json.dumps({"entry": "dn8_120", "gate": "dump_16_16"})) == (8, 120, 16, 16)
    assert plateau.key(json.dumps({"entry": "x"})) is None

def test_server_blocks_env_and_bad_window():
    class FakeEngine:
        def window(self, q): raise ValueError("bad window")
    serve.ENGINE = FakeEngine()
    srv = serve.ThreadingHTTPServer(("127.0.0.1", 0), serve.H); port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        for p, code in (("/.env", 404), ("/x/.env", 404), ("/api/window?preset=zz", 400)):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}{p}"); got = 200
            except urllib.error.HTTPError as e:
                got = e.code
            assert got == code, (p, got)
    finally:
        srv.shutdown()

def test_live_exit_value_matches_backtest_cost_formula():
    from solstrats.live import Position, Book, exit_value, venue_of
    from solstrats import costs as K
    p = Position("m", "X", 1.0, 10.0, 0.0, 1.0, venue=2, vol_entry=1000.0)
    pc = K.PCT_COST[2] + K.IMPACT_K * 10 / 1001; fix = K.FIXED_SOL * 80
    assert abs(exit_value(p, 2.0, 80) - ((10 - fix) * (1 - pc) * 2 * (1 - pc) - fix)) < 1e-12
    b = Book("s", 100.0, 90.0, positions={"m": p})
    assert b.liq_equity({"m": 2.0}, 80) < b.equity({"m": 2.0})          # mid mark ignores exit costs; liquidation mark does not
    assert venue_of({"dex": "pumpfun", "mint": "a"}, set()) == 2 and venue_of({"mint": "a"}, {"a"}) == 0 and venue_of({"mint": "b"}, {"a"}) == 1

def test_live_no_reentry_right_after_stop(tmp_path, monkeypatch):
    """A stopped-out token must not be bought again on the next poll (the backtest allows the next fill only 2 bars after exit)."""
    from solstrats import live as Lv
    px = {"v": 1.0}
    class PV:
        c = type("c", (), {"stats": {}})()
        def new_pairs(self, limit=60): return [{"mint": "M", "symbol": "EVIL", "price": px["v"], "dex": "raydium", "age_h": 1.0, "mcap": 5e4, "liq": 2e4, "vol_h1": 1e6, "chg_h1": 50}]
        def tokens(self, mints): return [dict(self.new_pairs()[0])]
        def prices(self, mints): return {mints[0]: 80.0}
        def mint_safety(self, m): return {}
        def status(self): return {}
    st = {"id": "D1", "family": "desk", "p": [0, 0, 0, 48], "e": [Lv.NONE, 0.1, 9, 0.5, 400, Lv.NONE, 1.0], "frac": 0.3, "maxpos": 3, "ovmask": 0}
    r = Lv.LiveRunner([st], [100.0], out_dir=str(tmp_path), providers=PV())
    monkeypatch.setattr(r, "refresh_hourly", lambda: None)
    r.poll(); bk = r.books["D1@100"]; assert "M" in bk.positions          # bought
    px["v"] = 0.5; r.poll(); assert "M" not in bk.positions and len(bk.closed) == 1   # stopped out
    px["v"] = 0.55; r.poll(); r.poll(); assert "M" not in bk.positions                # no instant re-entry
