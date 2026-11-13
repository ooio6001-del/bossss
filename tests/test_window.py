import numpy as np
from solstrats import engine as E, costs as K, window as W

def rand_data(T=400, N=4, seed=1):
    rng = np.random.default_rng(seed)
    C = np.exp(np.cumsum(rng.normal(0, 0.05, (T, N)), axis=0))
    O = np.r_[C[:1], C[:-1]]; H = np.maximum(O, C) * 1.02; L = np.minimum(O, C) * 0.98
    V = np.full((T, N), 1e7)
    sig = rng.random((T, N)) < 0.08
    D = {"O": O, "H": H, "L": L, "C": C, "V": V, "T": T, "last": np.full(N, T - 1), "supply": np.full(N, 1e9),
         "venue": np.zeros(N, np.int64), "solpx": np.full(T, 150.0), "times": 1.7e9 + 3600 * np.arange(T)}
    return D, sig

E1 = np.array([0.3, 0.15, 0.2, 0.1, 40, E.NONE, 1.0])

def gen(D, sig, one=False):
    return E.gen_trades(sig, E1, D["O"], D["H"], D["L"], D["C"], D["last"], D["supply"], one)

def test_replay_equals_masked_regeneration():
    D, sig = rand_data()
    for one in (False, True):
        cand = E.gen_trades_all(sig, E1, D["O"], D["H"], D["L"], D["C"], D["last"], D["supply"], one)
        for lo in (0, 1, 50, 199, 390):
            s = sig.copy(); s[: max(lo - 1, 0)] = False
            a = gen(D, s, one); b = W.replay(cand, lo, one)
            for x, y in zip(a[:3], b[:3]):
                assert np.array_equal(x, y)

def test_full_window_matches_engine_simulate():
    D, sig = rand_data(seed=3)
    tr = gen(D, sig)
    ve = D["V"][np.maximum(tr[1] - 1, 0), tr[0]]
    ref = E.simulate(*tr[:5], D["venue"], ve, D["solpx"], 0, D["T"], 100.0, 0.2, 5, K.PCT_COST, K.FIXED_SOL, K.RENT_SOL,
                     K.RENT_UNRECOVERED, K.IMPACT_K, K.FILL_CAP, np.zeros(0))
    # cut one bar before the data end so the engine's forced end-of-data exits are realized in both
    m = W.run(D, tr, 0, D["T"], 100.0, 0.2, 5)
    n_eo = int(W.end_open(D, tr).sum())
    assert int(m[3]) + int(m[6]) == int(ref[4])
    if n_eo == 0:
        assert abs(m[0] - ref[0]) < 1e-6

def test_open_positions_marked_to_window_close():
    D, sig = rand_data(seed=5)
    tr = gen(D, sig)
    hi = int(np.median(tr[2])) + 1
    lo = max(0, hi - 48)
    tok, ti, to, pin, pout, isopen = W.clip_to_window(D, tr, lo, hi)
    assert isopen.any() and (to < hi).all()
    k = np.nonzero(isopen)[0]
    assert np.allclose(pout[k], D["C"][hi - 1, tok[k]])
    m = W.run(D, tr, lo, hi, 100.0, 0.1, 10)
    assert m[6] == isopen.sum() or m[6] <= isopen.sum()   # some can be skipped by the position cap

def test_presets_and_capping():
    D, _ = rand_data(T=300)
    b = W.resolve(D, preset="24h"); assert b["hours"] == 24 and b["t_hi"] == 300
    b = W.resolve(D, preset="1h"); assert b["hours"] == 1
    b = W.resolve(D, preset="12m"); assert b["t_lo"] == 0 and b["capped_to_data"]
    b = W.resolve(D, start=float(D["times"][10]), end=float(D["times"][20])); assert (b["t_lo"], b["t_hi"]) == (10, 21)
