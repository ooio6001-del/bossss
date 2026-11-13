import numpy as np
from solstrats import engine as E, costs as K, grid as G, overlay as OV
from solstrats.live import self_checks

def mkdata(prices):
    C = np.array(prices, float)[:, None]
    O = np.r_[C[:1], C[:-1]]; H = np.maximum(O, C) * 1.0; L = np.minimum(O, C) * 1.0
    return O, H, L, C

def test_stop_loss_and_no_lookahead():
    O, H, L, C = mkdata([1, 1, 1, 1, 0.8, 0.8, 0.8])
    sig = np.zeros_like(C, bool); sig[1, 0] = True
    e = np.array([E.NONE, 0.1, E.NONE, 0, 100, E.NONE, 1.0])
    tok, ti, to, pin, pout, rs = E.gen_trades(sig, e, O, H, L, C, np.array([6]), np.array([1e9]), False)
    assert ti[0] == 2              # signal at bar 1 fills at bar 2 open
    assert rs[0] == E.R_SL and abs(pout[0] - 0.9) < 1e-9 or pout[0] <= 0.9

def test_take_profit():
    O, H, L, C = mkdata([1, 1, 1, 1.2, 1.5, 1.5])
    sig = np.zeros_like(C, bool); sig[0, 0] = True
    e = np.array([0.25, 0.5, E.NONE, 0, 100, E.NONE, 1.0])
    r = E.gen_trades(sig, e, O, H, L, C, np.array([5]), np.array([1e9]), True)
    assert r[5][0] == E.R_TP and r[4][0] >= 1.25

def test_trailing_stop():
    O, H, L, C = mkdata([1, 1, 1.5, 2.0, 1.7, 1.6])
    sig = np.zeros_like(C, bool); sig[0, 0] = True
    e = np.array([E.NONE, 0.5, 0.2, 0.1, 100, E.NONE, 1.0])
    r = E.gen_trades(sig, e, O, H, L, C, np.array([5]), np.array([1e9]), True)
    assert r[5][0] == E.R_TRAIL

def _sim(gross, bal, frac=1.0):
    tok = np.array([0]); ti = np.array([1]); to = np.array([2]); pin = np.array([1.0]); pout = np.array([gross])
    return E.simulate(tok, ti, to, pin, pout, np.array([0]), np.array([1e9]), np.full(10, 80.0), 0, 10, bal, frac, 1,
                      K.PCT_COST, K.FIXED_SOL, K.RENT_SOL, K.RENT_UNRECOVERED, K.IMPACT_K, K.FILL_CAP, np.zeros(0))

def test_flat_trade_loses_costs_and_small_accounts_pay_more():
    small = _sim(1.0, 5.0); big = _sim(1.0, 250.0)
    assert small[1] < 0 and big[1] < 0
    assert small[1] < big[1]   # fixed SOL fees + rent hurt $5 more than $250 in % terms

def test_profitable_trade():
    r = _sim(2.0, 100.0)
    assert 0.9 < r[1] < 1.0 and r[4] == 1 and r[5] == 1

def test_fill_cap_skips_illiquid():
    tok = np.array([0]); r = E.simulate(tok, np.array([1]), np.array([2]), np.array([1.0]), np.array([2.0]), np.array([0]), np.array([10.0]),
                                        np.full(10, 80.0), 0, 10, 100.0, 1.0, 1, K.PCT_COST, K.FIXED_SOL, K.RENT_SOL, K.RENT_UNRECOVERED,
                                        K.IMPACT_K, K.FILL_CAP, np.zeros(0))
    assert r[4] == 0 and r[10] == 1 and r[0] == 100.0

def test_space_and_grid_sizes():
    assert G.space_size() >= 5e40
    n = sum(len(G.EXIT_SETS[e[2]]) for e in G.entries()) * len(G.SIZING) * len(K.BALANCES)
    assert n >= 4_000_000
    assert 280.0 in K.BALANCES and K.BALANCES[0] == 5.0

def test_overlay_masks():
    sig = np.ones((3, 2), bool); chk = np.zeros((8, 3, 2), bool); chk[1, :, 0] = True
    assert OV.apply_checks(sig, chk, 0).all()
    s = OV.apply_checks(sig, chk, 2)
    assert s[:, 0].all() and not s[:, 1].any()
    assert OV.risk_exit((E.NONE, 0.2, E.NONE, 0, 24, E.NONE, 1.0))[1:4] == (0.06, 0.12, 0.06)

def test_self_checks_unknowns_never_faked():
    qa = self_checks({"age_h": 2, "liq": 50000, "vol_h1": 60000, "chg_h1": 5, "buys_h1": 10, "sells_h1": 5}, {}, {})
    d = {q["q"]: q["a"] for q in qa}
    assert d["LP burned/locked?"] == "unknown" and d["Dev wallet clean (no past rugs)?"] == "unknown"
    assert d["Mint authority revoked?"] == "unknown"

def test_wick_haircut_blocks_single_print_tp():
    # bar 3 has a lone upper wick to 1.5 but closes at 1.0: a +40% TP should NOT fill (only half the wick counts)
    C = np.array([1, 1, 1, 1.0, 1.0, 1.0])[:, None]; O = C.copy(); L = C.copy(); H = C.copy(); H[3, 0] = 1.5
    sig = np.zeros_like(C, bool); sig[0, 0] = True
    e = np.array([0.4, 0.5, E.NONE, 0, 100, E.NONE, 1.0])
    r = E.gen_trades(sig, e, O, H, L, C, np.array([5]), np.array([1e9]), True)
    assert r[5][0] != E.R_TP

def test_fixed_micro_size_fees_dominate():
    tok = np.array([0]); args = (np.array([0]), np.array([1e9]), np.full(10, 80.0), 0, 10, 100.0)
    tiny = E.simulate(tok, np.array([1]), np.array([2]), np.array([1.0]), np.array([1.05]), *args, -0.01, 1, K.PCT_COST, K.FIXED_SOL,
                      K.RENT_SOL, K.RENT_UNRECOVERED, K.IMPACT_K, K.FILL_CAP, np.zeros(0))
    big = E.simulate(tok, np.array([1]), np.array([2]), np.array([1.0]), np.array([1.05]), *args, -5.0, 1, K.PCT_COST, K.FIXED_SOL,
                     K.RENT_SOL, K.RENT_UNRECOVERED, K.IMPACT_K, K.FILL_CAP, np.zeros(0))
    assert tiny[4] == 1 and tiny[0] < 100.0      # a +5% trade on $0.01 still loses money to fixed SOL costs
    assert big[0] > 100.0                         # same trade at $5 is profitable
