"""Brute-force grid definition + nominal search-space size."""
from __future__ import annotations
import itertools, math
import numpy as np
from .engine import F, NONE

def _exits(tps, sls, trails, tmaxs, mcts=(NONE,), crashes=(1.0,)):
    out = []
    for tp, sl, (act, w), tm, mct, cr in itertools.product(tps, sls, trails, tmaxs, mcts, crashes):
        out.append((tp, sl, act, w, tm, mct, cr))
    return out

NT = (NONE, 0.0)
E_STD = _exits((0.1, 0.25, 0.5, 1.0, NONE), (0.05, 0.1, 0.2, 0.35), (NT, (0.1, 0.1), (0.25, 0.2)), (24, 72, 240))
E_SCALP = _exits((0.02, 0.03, 0.05), (0.15, 0.3, 0.5), (NT,), (12, 48))            # high-win-rate objective
E_MOON = _exits((2.5, 3, 4, 5, 7.5, 10, 15, 20, 30, 50, 75, 100), (0.5, 0.7, 1.0), (NT, (2.0, 0.5)), (240, 720), crashes=(1.0, 0.4))
E_DESK = [(NONE, sl, act, w, tm, NONE, 1.0) for sl in (0.03, 0.06, 0.1, 0.15) for act in (0.05, 0.12, 0.25, 0.5)
          for w in (0.03, 0.06, 0.12) for tm in (24, 168)]
E_MC = ([(NONE, sl, a, w, tm, mct, 1.0) for mct in (1e5, 5e5, 1e6, 3e6) for sl in (0.3, 0.5) for a, w in (NT, (0.5, 0.3)) for tm in (24, 168)] +
        [(tp, sl, a, w, tm, NONE, 1.0) for tp in (1.0, 3.0) for sl in (0.3, 0.5) for a, w in (NT, (0.5, 0.3)) for tm in (24, 168)])

def entries():
    """(family, params[6], exit_set_name, one_shot)"""
    E = []
    P = lambda *a: tuple(list(a) + [0.0] * (6 - len(a)))
    for L, th, vm in itertools.product((24, 72, 168), (0.0, 0.05), (0, 3)):
        E.append(("breakout", P(L, th, vm), "std+scalp", False))
    for fa, sl, vm in itertools.product((6, 24), (72, 168), (0, 3)):
        E.append(("ma_cross", P(fa, sl, vm), "std+scalp", False))
    for per, thr, tr in itertools.product((6, 14), (20, 30), (0, 1)):
        E.append(("rsi_revert", P(per, thr, tr), "std+scalp", False))
    for vm, r in itertools.product((3, 6, 10), (0.05, 0.15)):
        E.append(("vol_spike", P(vm, r), "std+scalp", False))
    for L, d, b in itertools.product((24, 72), (0.3, 0.5), (0, 1)):
        E.append(("dip_buy", P(L, d, b), "std+scalp", False))
    for age, mc, vol, mom in itertools.product((6, 24, 72), (2e5, NONE), (1e3, 1e4), (0.0, 0.2)):
        E.append(("moonshot", P(age, mc, vol, mom), "moon", True))
    for thr, t10, vol in itertools.product((0.4, 0.5, 0.6, 0.7, 0.8, 0.9), (0.2, 0.3, 0.5), (5e3, 2.5e4, 5e4)):
        E.append(("desk", P(thr, t10, vol, 720, 63), "desk", False))
    # question-ablation: every include/exclude subset of the 6 desk questions at the post's default cutoffs
    for mask in range(64):
        if mask != 63:
            E.append(("desk", P(0.6, 0.2, 5e4, 720, mask), "desk_post", False))
    for mx, g, n, vc in itertools.product((2.5e4, 1e5, 5e5), (0.5, 1.0, 2.0), (1, 3), (0, 3)):
        E.append(("fast_mc", P(2500, mx, g, n, vc), "mc", False))
    for age, mom, vol in itertools.product((1, 3), (-0.2, 0.0, 0.2), (0, 5e3)):
        E.append(("graduation", P(age, mom, vol), "std", True))
    for mcx, vol in itertools.product((1e4, 2e4, 3e4), (0, 1e3)):
        E.append(("bonding_curve", P(mcx, vol), "mc", True))
    for L, mom in itertools.product((24, 72, 168), (0.0, 0.1)):
        E.append(("narrative", P(L, mom), "std", False))
    for d, vm, fr in itertools.product((0.1, 0.2, 0.35), (0, 3), (0, 0.8)):
        E.append(("flash_wick", P(d, vm, fr), "std+scalp", False))
    # Extreme Upside: very early entries, uncapped runners
    for mx, g_, n in itertools.product((5e3, 1e4, 3e4), (0.2, 0.5, 1.0), (1, 3)):
        E.append(("fast_mc", P(2500, mx, g_, n, 0), "runner", False))
    for age, mc in itertools.product((3, 24, 168), (5e3, 3e4, 1e5, NONE)):
        E.append(("moonshot", P(age, mc, 0, 0), "runner", True))
    for mcx in (3e3, 5e3, 1e4):
        E.append(("bonding_curve", P(mcx, 0), "runner", True))
    for L in (24, 72):
        E.append(("breakout", P(L, 0.0, 3), "runner", False))
    return E

E_RUNNER = [(NONE, sl, act, w, tm, NONE, 1.0) for sl in (0.5, 0.8, 1.0) for act, w in ((1.0, 0.5), (3.0, 0.6), (10.0, 0.7))
            for tm in (720, 2160)]   # "Extreme Upside": uncapped runners, wide trailing stop only
E_DESK_POST = [(NONE, 0.06, 0.12, w, tm, NONE, 1.0) for w in (0.03, 0.06) for tm in (24, 168)]
EXIT_SETS = {"runner": E_RUNNER, "desk_post": E_DESK_POST, "std": E_STD, "std+scalp": E_STD + E_SCALP, "moon": E_MOON, "desk": E_DESK, "mc": E_MC}
SIZING = [(0.08, 12), (0.2, 5), (0.34, 3), (0.5, 2), (1.0, 1), (0.02, 50), (0.05, 20)]   # last two: lottery-style many tiny bets   # (fraction of equity per trade, max open positions)

# ---- nominal combinatorial search space for the smart (GA / random) search -------------------------
# Every parameter is discretized at the resolution the optimizer samples at; the space is the product.
SPACE = {
    # family-specific entry parameters: (low, high, step)
    "breakout": {"lookback_h": (2, 720, 1), "threshold": (0, 0.5, 0.0005), "vol_mult": (0, 20, 0.01)},
    "ma_cross": {"fast_h": (2, 96, 1), "slow_h": (24, 720, 1), "vol_mult": (0, 20, 0.01)},
    "rsi_revert": {"period_h": (2, 72, 1), "rsi_below": (2, 50, 0.01), "trend_filter": (0, 1, 1)},
    "vol_spike": {"vol_mult": (1.5, 50, 0.01), "min_ret": (0, 1, 0.0005)},
    "dip_buy": {"lookback_h": (6, 720, 1), "drawdown": (0.05, 0.95, 0.0005), "bounce": (0, 1, 1)},
    "moonshot": {"max_age_h": (1, 720, 1), "max_mcap": (5e3, 5e7, 1e3), "min_vol_h": (0, 1e6, 100), "min_ret_h": (-0.5, 3, 0.001)},
    "desk": {"question_mask": (0, 63, 1), "score_min": (0.3, 0.95, 0.001), "top10_max": (0.05, 0.9, 0.001), "min_vol_h": (0, 1e6, 100), "max_age_h": (6, 2160, 1)},
    "fast_mc": {"mc_min": (2500, 1e5, 100), "mc_max": (1e4, 3e6, 1e3), "growth": (0.1, 10, 0.001), "window_h": (1, 48, 1), "vol_mult": (0, 20, 0.01)},
    "graduation": {"max_age_h": (1, 72, 1), "min_ret": (-0.9, 5, 0.001), "min_vol_h": (0, 1e6, 100)},
    "bonding_curve": {"cross_mcap": (3e3, 6.9e4, 100), "min_vol_h": (0, 1e5, 10)},
    "flash_wick": {"dump": (0.03, 0.9, 0.001), "vol_mult": (0, 20, 0.01), "no_freefall": (0, 1, 0.01)},
    "narrative": {"lookback_h": (6, 720, 1), "min_ret": (-0.5, 2, 0.001)},
}
EXIT_SPACE = {"take_profit": (0.005, 50, 0.005), "stop_loss": (0.01, 1.0, 0.001), "trail_activate": (0.01, 20, 0.005),
              "trail_width": (0.01, 0.9, 0.001), "time_stop_h": (1, 2160, 1), "mcap_target": (1e4, 3e6, 1e3),
              "crash_exit": (0.05, 1.0, 0.001)}
SIZE_SPACE = {"overlay_mask": (0, 255, 1), "risk_agent": (0, 1, 1), "executor_agent": (0, 1, 1), "fraction": (0.01, 1.0, 0.001), "max_positions": (1, 50, 1), "balance": (5, 280, 5)}

def _levels(lo, hi, step):
    return int(round((hi - lo) / step)) + 1

def space_size() -> float:
    shared = 1.0
    for v in list(EXIT_SPACE.values()) + list(SIZE_SPACE.values()):
        shared *= _levels(*v)
    total = 0.0
    for fam, ps in SPACE.items():
        s = 1.0
        for v in ps.values():
            s *= _levels(*v)
        total += s * shared
    return total
