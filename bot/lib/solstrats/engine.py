"""Vectorized/numba backtest engine.

Stage 1  signals(fam, p)            -> bool[T, N]   (entry signal at bar close t; fill at open t+1)
Stage 2  gen_trades(sig, exits)     -> per-token trade list (one position per token at a time)
Stage 3  simulate(trades, balance, sizing, costs) -> portfolio metrics (small-account cost model)
No look-ahead: signals use bars <= t; entries fill at the next bar's open; stops/targets use later bars.
If a stop and a target are both touched inside one hourly bar, the stop is assumed to fill first."""
from __future__ import annotations
import math
import numpy as np
from numba import njit, prange

FAMILIES = ["breakout", "ma_cross", "rsi_revert", "vol_spike", "dip_buy", "moonshot", "desk", "fast_mc",
            "graduation", "bonding_curve", "narrative", "flash_wick"]
F = {n: i for i, n in enumerate(FAMILIES)}
# exit vector layout
E_TP, E_SL, E_ACT, E_W, E_TMAX, E_MCT, E_CRASH = range(7)
NONE = 1e9
WICK_FILL = 0.5   # share of an upper wick a take-profit order can realistically fill against
# reasons
R_SL, R_TP, R_TRAIL, R_TIME, R_MCT, R_CRASH, R_END = range(7)
REASONS = ["stop-loss", "take-profit", "trailing-stop", "time-stop", "mcap-target", "crash-exit", "data-end"]

@njit(cache=True)
def _clip01(x):
    return 0.0 if x < 0 else (1.0 if x > 1 else x)

@njit(cache=True)
def desk_score(C, V, t, j, first, top10j):
    mom = C[t, j] / C[t - 6, j] - 1.0 if t >= 6 and C[t - 6, j] > 0 else 0.0
    s1 = _clip01((mom + 0.2) / 0.7)
    acc = 0.0; n = 0
    for k in range(max(first, t - 24), t):
        acc += V[k, j]; n += 1
    avg = acc / n if n > 0 else 0.0
    vr = V[t, j] / avg if avg > 0 else 1.0
    s2 = _clip01((vr - 1.0) / 4.0)
    s3 = _clip01(1.0 - (t - first) / 720.0)
    conc = top10j if not math.isnan(top10j) else 0.6
    s4 = _clip01(1.0 - conc / 0.6)
    s5 = _clip01((math.log10(V[t, j] + 1.0) - 3.0) / 2.5)
    return (s1 + s2 + s3 + s4 + s5) / 5.0

@njit(cache=True, parallel=True)
def signals(fam, p, O, H, L, C, V, first, grp, venue, supply, mint_auth, freeze, top10, hot, theme, t_lo, t_hi):
    T, N = C.shape
    sig = np.zeros((T, N), dtype=np.bool_)
    for j in prange(N):
        if grp[j] == 2:
            continue  # SOL benchmark never traded
        f0 = first[j]
        if f0 < 0:
            continue
        launch = grp[j] == 1
        if fam in (5, 6, 8) and not launch:
            continue
        if fam == 9 and venue[j] != 2:
            continue
        if fam == 6:
            mask = int(p[4])
            if (mask & 1) and mint_auth[j] > 0:      # FILTER Q: mint authority revoked?
                continue
            if (mask & 2) and freeze[j] > 0:         # FILTER Q: freeze authority revoked?
                continue
            if (mask & 4) and (math.isnan(top10[j]) or top10[j] > p[1]):   # FILTER Q: top-10 holders under cutoff?
                continue
        # rolling helpers
        for t in range(max(f0 + 1, t_lo), t_hi):
            c = C[t, j]
            if math.isnan(c) or c <= 0:
                continue
            if fam == 0:  # breakout: close > max(high[t-L..t-1])*(1+th), volume >= vm * avg24
                Lb = int(p[0])
                if t - f0 < Lb: continue
                mx = 0.0
                for k in range(t - Lb, t):
                    if H[k, j] > mx: mx = H[k, j]
                if c <= mx * (1 + p[1]): continue
                if p[2] > 0:
                    a = 0.0
                    for k in range(t - 24, t): a += V[k, j]
                    if V[t, j] < p[2] * a / 24.0: continue
                sig[t, j] = True
            elif fam == 1:  # MA cross (fast crosses above slow)
                fa, sl = int(p[0]), int(p[1])
                if t - f0 < sl + 1: continue
                f1 = 0.0; s1 = 0.0; f2 = 0.0; s2 = 0.0
                for k in range(t - fa + 1, t + 1): f1 += C[k, j]
                for k in range(t - sl + 1, t + 1): s1 += C[k, j]
                for k in range(t - fa, t): f2 += C[k, j]
                for k in range(t - sl, t): s2 += C[k, j]
                if f1 / fa > s1 / sl and f2 / fa <= s2 / sl:
                    if p[2] > 0:
                        a = 0.0
                        for k in range(t - 24, t): a += V[k, j]
                        if V[t, j] < p[2] * a / 24.0: continue
                    sig[t, j] = True
            elif fam == 2:  # RSI mean reversion, optional uptrend filter (close > MA168)
                per = int(p[0])
                if t - f0 < max(per + 1, 170 if p[2] > 0 else 0): continue
                g = 0.0; l = 0.0
                for k in range(t - per + 1, t + 1):
                    d = C[k, j] - C[k - 1, j]
                    if d > 0: g += d
                    else: l -= d
                rsi = 100.0 if l == 0 else 100.0 - 100.0 / (1.0 + g / l)
                if rsi >= p[1]: continue
                if p[2] > 0:
                    m = 0.0
                    for k in range(t - 167, t + 1): m += C[k, j]
                    if c < m / 168.0: continue
                sig[t, j] = True
            elif fam == 3:  # volume spike with green candle
                if t - f0 < 25: continue
                a = 0.0
                for k in range(t - 24, t): a += V[k, j]
                a /= 24.0
                if a <= 0 or V[t, j] < p[0] * a: continue
                if c / C[t - 1, j] - 1.0 < p[1]: continue
                sig[t, j] = True
            elif fam == 4:  # dip buy: drawdown from L-high >= d, optional bounce (green bar)
                Lb = int(p[0])
                if t - f0 < Lb: continue
                mx = 0.0
                for k in range(t - Lb, t + 1):
                    if H[k, j] > mx: mx = H[k, j]
                if c > mx * (1 - p[1]): continue
                if p[2] > 0 and c <= C[t - 1, j]: continue
                sig[t, j] = True
            elif fam == 5:  # moonshot: young launch, low mcap, volume + momentum
                age = t - f0
                if age > p[0]: break
                if c * supply[j] > p[1]: continue
                if V[t, j] < p[2]: continue
                if c / C[t - 1, j] - 1.0 < p[3]: continue
                sig[t, j] = True
            elif fam == 6:  # desk: SCOUT (new pair?), FILTER (mask questions), ANALYST (score gate)
                mask = int(p[4])
                age = t - f0
                if age > p[3]: break                                   # SCOUT Q: still a new pair?
                if (mask & 8) and V[t, j] < p[2]: continue             # FILTER Q: hourly volume above cutoff?
                if t < 7: continue
                if (mask & 16) and C[t, j] <= C[t - 6, j]: continue     # ANALYST Q: momentum rising (6h)?
                if (mask & 32) and desk_score(C, V, t, j, f0, top10[j]) < p[0]: continue   # ANALYST Q: score >= gate?
                sig[t, j] = True
            elif fam == 7:  # fast mcap movers: mc in [p0, p1], grew >= p2 over p3 hours, vol confirm p4
                n = int(p[3])
                if t - f0 < max(n, 24): continue
                mc = c * supply[j]
                if mc < p[0] or mc > p[1]: continue
                prev = C[t - n, j]
                if prev <= 0 or c / prev - 1.0 < p[2]: continue
                if p[4] > 0:
                    a = 0.0
                    for k in range(t - 24, t): a += V[k, j]
                    if V[t, j] < p[4] * a / 24.0: continue
                sig[t, j] = True
            elif fam == 8:  # graduation play: first hours of a new pool
                age = t - f0
                if age > p[0]: break
                if c / O[f0, j] - 1.0 < p[1]: continue
                if V[t, j] < p[2]: continue
                sig[t, j] = True
            elif fam == 9:  # bonding curve: pump.fun curve token crossing mcap p0 (below graduation)
                mc = c * supply[j]
                pm = C[t - 1, j] * supply[j]
                if mc >= p[0] and pm < p[0] and V[t, j] >= p[1]:
                    sig[t, j] = True
            elif fam == 10:  # narrative rotation: token in the hottest theme, own momentum >= p1
                Lb = int(p[0])
                if t - f0 < Lb: continue
                if hot[t] != theme[j]: continue
                if (t % 24) != 0: continue  # rebalance daily
                if c / C[t - Lb, j] - 1.0 < p[1]: continue
                sig[t, j] = True
            elif fam == 11:  # flash-wick mean reversion: single-hour dump >= p0 (close vs prior close), vol >= p1 x avg,
                #             and prior 24h not already in freefall (close[t-1] >= p2 x close[t-24])
                if t - f0 < 25: continue
                pc = C[t - 1, j]
                if pc <= 0 or c / pc - 1.0 > -p[0]: continue
                if p[1] > 0:
                    a = 0.0
                    for k in range(t - 24, t): a += V[k, j]
                    if V[t, j] < p[1] * a / 24.0: continue
                if p[2] > 0 and pc < p[2] * C[t - 24, j]: continue
                sig[t, j] = True
    return sig

@njit(cache=True)
def gen_trades(sig, e, O, H, L, C, last, supply, one_shot):
    """Returns arrays: token, t_entry(bar of fill), t_exit, px_in, px_out, reason."""
    T, N = C.shape
    cap = 1 << 14
    tok = np.empty(cap, np.int64); ti = np.empty(cap, np.int64); to = np.empty(cap, np.int64)
    pin = np.empty(cap); pout = np.empty(cap); rs = np.empty(cap, np.int64)
    n = 0
    tp, sl, act, w, tmax, mct, crash = e[0], e[1], e[2], e[3], int(e[4]), e[5], e[6]
    for j in range(N):
        t = 0
        lj = last[j]
        while t < lj:
            if not sig[t, j]:
                t += 1; continue
            b = t + 1
            px = O[b, j]
            if math.isnan(px) or px <= 0:
                t += 1; continue
            stop = px * (1 - sl) if sl < 1 else 0.0
            tgt = px * (1 + tp) if tp < NONE / 2 else NONE
            mct_px = mct / supply[j] if mct < NONE / 2 else NONE
            hw = px; active = False
            k = b; xp = 0.0; r = R_END
            while True:
                st = stop
                if active:
                    tr = hw * (1 - w)
                    if tr > st: st = tr
                if L[k, j] <= st:
                    xp = min(st, O[k, j]) if k > b else st
                    r = R_TRAIL if (active and st > stop) else R_SL
                    break
                # upper wicks on thin pools are often single small trades: count only half the wick as fillable
                bodyhi = O[k, j] if O[k, j] > C[k, j] else C[k, j]
                hx = bodyhi + WICK_FILL * (H[k, j] - bodyhi)
                if hx >= tgt:
                    xp = max(tgt, O[k, j]) if k > b else tgt
                    r = R_TP; break
                if hx >= mct_px:
                    xp = max(mct_px, O[k, j]) if k > b else mct_px
                    r = R_MCT; break
                if crash < 1 and k > b and C[k, j] / C[k - 1, j] - 1.0 <= -crash:
                    xp = C[k, j]; r = R_CRASH; break
                if k - b >= tmax:
                    xp = C[k, j]; r = R_TIME; break
                if k >= lj:
                    xp = C[k, j]; r = R_END; break
                if H[k, j] > hw: hw = H[k, j]
                if not active and hw >= px * (1 + act): active = True
                k += 1
            if n >= cap:
                cap *= 2
                tok = np.concatenate((tok, np.empty(cap - tok.size, np.int64))); ti = np.concatenate((ti, np.empty(cap - ti.size, np.int64)))
                to = np.concatenate((to, np.empty(cap - to.size, np.int64))); pin = np.concatenate((pin, np.empty(cap - pin.size)))
                pout = np.concatenate((pout, np.empty(cap - pout.size))); rs = np.concatenate((rs, np.empty(cap - rs.size, np.int64)))
            tok[n] = j; ti[n] = b; to[n] = k; pin[n] = px; pout[n] = xp; rs[n] = r; n += 1
            if one_shot:
                break
            t = k + 1
    return tok[:n], ti[:n], to[:n], pin[:n], pout[:n], rs[:n]

@njit(cache=True)
def gen_trades_all(sig, e, O, H, L, C, last, supply, one_shot):
    """Like gen_trades but opens a candidate trade at EVERY signal bar (no position overlap rule). Used by the
    date-range picker: replaying these candidates with the overlap rule from any window start reproduces gen_trades on
    a window-masked signal exactly."""
    T, N = C.shape
    cap = 1 << 14
    tok = np.empty(cap, np.int64); ti = np.empty(cap, np.int64); to = np.empty(cap, np.int64)
    pin = np.empty(cap); pout = np.empty(cap); rs = np.empty(cap, np.int64)
    n = 0
    tp, sl, act, w, tmax, mct, crash = e[0], e[1], e[2], e[3], int(e[4]), e[5], e[6]
    for j in range(N):
        t = 0
        lj = last[j]
        while t < lj:
            if not sig[t, j]:
                t += 1; continue
            b = t + 1
            px = O[b, j]
            if math.isnan(px) or px <= 0:
                t += 1; continue
            stop = px * (1 - sl) if sl < 1 else 0.0
            tgt = px * (1 + tp) if tp < NONE / 2 else NONE
            mct_px = mct / supply[j] if mct < NONE / 2 else NONE
            hw = px; active = False
            k = b; xp = 0.0; r = R_END
            while True:
                st = stop
                if active:
                    tr = hw * (1 - w)
                    if tr > st: st = tr
                if L[k, j] <= st:
                    xp = min(st, O[k, j]) if k > b else st
                    r = R_TRAIL if (active and st > stop) else R_SL
                    break
                # upper wicks on thin pools are often single small trades: count only half the wick as fillable
                bodyhi = O[k, j] if O[k, j] > C[k, j] else C[k, j]
                hx = bodyhi + WICK_FILL * (H[k, j] - bodyhi)
                if hx >= tgt:
                    xp = max(tgt, O[k, j]) if k > b else tgt
                    r = R_TP; break
                if hx >= mct_px:
                    xp = max(mct_px, O[k, j]) if k > b else mct_px
                    r = R_MCT; break
                if crash < 1 and k > b and C[k, j] / C[k - 1, j] - 1.0 <= -crash:
                    xp = C[k, j]; r = R_CRASH; break
                if k - b >= tmax:
                    xp = C[k, j]; r = R_TIME; break
                if k >= lj:
                    xp = C[k, j]; r = R_END; break
                if H[k, j] > hw: hw = H[k, j]
                if not active and hw >= px * (1 + act): active = True
                k += 1
            if n >= cap:
                cap *= 2
                tok = np.concatenate((tok, np.empty(cap - tok.size, np.int64))); ti = np.concatenate((ti, np.empty(cap - ti.size, np.int64)))
                to = np.concatenate((to, np.empty(cap - to.size, np.int64))); pin = np.concatenate((pin, np.empty(cap - pin.size)))
                pout = np.concatenate((pout, np.empty(cap - pout.size))); rs = np.concatenate((rs, np.empty(cap - rs.size, np.int64)))
            tok[n] = j; ti[n] = b; to[n] = k; pin[n] = px; pout[n] = xp; rs[n] = r; n += 1
            t += 1
    return tok[:n], ti[:n], to[:n], pin[:n], pout[:n], rs[:n]

# metrics layout
M_NAMES = ["final", "ret", "maxdd", "sharpe", "ntrades", "wins", "gross_win", "gross_loss", "best", "worst",
           "skipped", "fees", "pf", "winrate", "avg_win", "avg_loss"]
NM = len(M_NAMES)

@njit(cache=True)
def simulate(tok, ti, to, pin, pout, venue_of_tok, vol_entry, solpx, t_lo, t_hi, balance, frac, maxpos,
             pct_cost, fixed_sol, rent_sol, rent_unrec, impact_k, fill_cap, curve_out):
    """Small-account portfolio simulation over trades whose entry bar is in [t_lo, t_hi).
    Positions are sized as frac * equity (marked at cost); at most maxpos open at once.
    Per trade costs: pct_cost[venue] per side (DEX fee + slippage) + price impact impact_k*size/hourly_vol,
    fixed SOL fees per swap (base + priority), token-account rent (locked, rent_unrec fraction lost).
    A trade is skipped when size > fill_cap * entry-bar USD volume (cannot fill) or cash is short."""
    out = np.zeros(NM)
    n = tok.size
    order = np.argsort(ti * 1.0 + tok * 1e-7)
    cash = balance
    op_exit = np.full(maxpos, -1, np.int64); op_val = np.zeros(maxpos); op_cost = np.zeros(maxpos); op_lock = np.zeros(maxpos)
    nopen = 0
    peak = balance; maxdd = 0.0
    ndays = (t_hi - t_lo) // 24 + 1
    daily = np.full(ndays, np.nan)
    ntr = 0; wins = 0; gw = 0.0; gl = 0.0; best = -1e18; worst = 1e18; skipped = 0; fees = 0.0
    last_day = -1
    eq = balance
    for ii in range(n + 1):
        if ii < n:
            q = order[ii]
            t_now = ti[q]
            if t_now < t_lo or t_now >= t_hi:
                continue
        else:
            t_now = 1 << 40
        # close positions that exited before t_now (in exit order)
        while True:
            mi = -1; me = 1 << 41
            for s in range(maxpos):
                if op_exit[s] >= 0 and op_exit[s] < t_now and op_exit[s] < me:
                    me = op_exit[s]; mi = s
            if mi < 0: break
            cash += op_val[mi] + op_lock[mi] * (1 - rent_unrec)
            pnl = op_val[mi] - op_cost[mi] - op_lock[mi] * rent_unrec
            ntr += 1
            if pnl > 0: wins += 1; gw += pnl
            else: gl -= pnl
            if pnl > best: best = pnl
            if pnl < worst: worst = pnl
            op_exit[mi] = -1; nopen -= 1
            eq = cash
            for s in range(maxpos):
                if op_exit[s] >= 0: eq += op_cost[s] + op_lock[s]
            if eq > peak: peak = eq
            dd = 1 - eq / peak if peak > 0 else 0.0
            if dd > maxdd: maxdd = dd
            d = (me - t_lo) // 24
            if d >= 0 and d < ndays: daily[d] = eq
            if curve_out.shape[0] > 0 and me - t_lo < curve_out.shape[0] and me >= t_lo:
                curve_out[me - t_lo] = eq
        if ii >= n:
            break
        if nopen >= maxpos:
            skipped += 1; continue
        eq = cash
        for s in range(maxpos):
            if op_exit[s] >= 0: eq += op_cost[s] + op_lock[s]
        fixed_mode = frac < 0              # negative frac = fixed USD buy size (micro-size study)
        size = -frac if fixed_mode else frac * eq
        sp = solpx[t_now]
        fix_usd = fixed_sol * sp
        lock_usd = rent_sol * sp
        if size + lock_usd > cash:
            if fixed_mode:
                skipped += 1; continue
            size = cash - lock_usd          # keep SOL for the token-account rent
        if size <= (0.0 if fixed_mode else fix_usd * 2):
            skipped += 1; continue
        ve = vol_entry[q]
        if size > fill_cap * ve:
            skipped += 1; continue
        imp = impact_k * size / (ve + 1.0)
        pc = pct_cost[venue_of_tok[tok[q]]] + imp
        gross = pout[q] / pin[q]
        val = (size - fix_usd) * (1 - pc) * gross * (1 - pc) - fix_usd
        fees += size * pc * (1 + gross) + 2 * fix_usd + rent_unrec * lock_usd
        cash -= size + lock_usd
        for s in range(maxpos):
            if op_exit[s] < 0:
                op_exit[s] = to[q]; op_val[s] = val; op_cost[s] = size; op_lock[s] = lock_usd
                break
        nopen += 1
    # daily equity -> sharpe
    prev = balance; cnt = 0; s1 = 0.0; s2 = 0.0
    for d in range(ndays):
        v = daily[d] if not math.isnan(daily[d]) else prev
        r = v / prev - 1.0 if prev > 0 else 0.0
        s1 += r; s2 += r * r; cnt += 1; prev = v
    mu = s1 / cnt; sd = math.sqrt(max(s2 / cnt - mu * mu, 0.0))
    out[0] = cash; out[1] = cash / balance - 1.0; out[2] = maxdd
    out[3] = mu / sd * math.sqrt(365.0) if sd > 0 else 0.0
    out[4] = ntr; out[5] = wins; out[6] = gw; out[7] = gl
    out[8] = best if ntr > 0 else 0.0; out[9] = worst if ntr > 0 else 0.0
    out[10] = skipped; out[11] = fees
    out[12] = gw / gl if gl > 0 else (99.0 if gw > 0 else 0.0)
    out[13] = wins / ntr if ntr > 0 else 0.0
    out[14] = gw / wins if wins > 0 else 0.0
    out[15] = gl / (ntr - wins) if ntr - wins > 0 else 0.0
    return out

@njit(cache=True, parallel=True)
def simulate_many(tok, ti, to, pin, pout, venue_of_tok, vol_entry, solpx, offs, cfg_combo, cfg_bal, cfg_frac, cfg_maxpos,
                  t_lo, t_hi, pct_cost, fixed_sol, rent_sol, rent_unrec, impact_k, fill_cap):
    M = cfg_combo.size
    res = np.zeros((M, NM), dtype=np.float32)
    empty = np.zeros(0)
    for m in prange(M):
        c = cfg_combo[m]
        a, b = offs[c], offs[c + 1]
        r = simulate(tok[a:b], ti[a:b], to[a:b], pin[a:b], pout[a:b], venue_of_tok, vol_entry[a:b], solpx, t_lo, t_hi,
                     cfg_bal[m], cfg_frac[m], cfg_maxpos[m], pct_cost, fixed_sol, rent_sol, rent_unrec, impact_k, fill_cap, empty)
        for k in range(NM):
            res[m, k] = r[k]
    return res
