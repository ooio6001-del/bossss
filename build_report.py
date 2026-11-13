"""Aggregate sweep + overlay + smart-search results into report.json and CSVs (no fabricated numbers)."""
from __future__ import annotations
import csv, json, math, os, sys, time
import numpy as np
from solstrats import engine as E, costs as K, grid as G, overlay as OV, data as Dm, walkforward as WF
from solstrats.describe import entry_text, exit_text, sizing_text
from run_backtest import prep

M = {n: i for i, n in enumerate(E.M_NAMES)}
R = sys.argv[1] if len(sys.argv) > 1 else "results"
iso = lambda t: time.strftime("%Y-%m-%d", time.gmtime(t))

def r4(x): return None if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else round(float(x), 4)

def bh(D, lo, hi, cols):
    """equal-weight buy & hold return of cols between bars lo..hi-1 (tokens alive at lo), with one round-trip cost."""
    C = D["C"]; rets = []
    for j in cols:
        a, b = C[lo, j], C[hi - 1, j]
        if not np.isnan(a) and not np.isnan(b) and a > 0:
            rets.append(b / a * (1 - K.PCT_COST[min(D["venue"][j], 1)]) ** 2 - 1)
    return float(np.mean(rets)) if rets else None, len(rets)

def resim(D, cb, frac, maxpos, bal, lo, hi, mask=0, risk=0, fill_cap=K.FILL_CAP, chk=None):
    fam = cb["family"]; p = np.asarray(cb["p"], float)
    hot = D["hot"].get(int(p[0]), D["hot"][24]) if fam == "narrative" else D["hot"][24]
    sig = E.signals(E.F[fam], p, D["O"], D["H"], D["L"], D["C"], D["V"], D["first"], D["grp"], D["venue"], D["supply"],
                    D["mint_auth"], D["freeze"], D["top10"], hot, D["theme"], 0, D["T"] - 1)
    if mask: sig = OV.apply_checks(sig, chk, mask)
    e = OV.risk_exit(tuple(cb["e"])) if risk else tuple(cb["e"])
    tr = E.gen_trades(sig, np.asarray(e, float), D["O"], D["H"], D["L"], D["C"], D["last"], D["supply"], cb["one_shot"])
    ve = D["V"][np.maximum(tr[1] - 1, 0), tr[0]]
    curve = np.full(hi - lo, np.nan)
    m = E.simulate(*tr[:5], D["venue"], ve, D["solpx"], lo, hi, bal, frac, maxpos, K.PCT_COST, K.FIXED_SOL, K.RENT_SOL,
                   K.RENT_UNRECOVERED, K.IMPACT_K, fill_cap, curve)
    # forward-fill curve
    v = bal; out = []
    for x in curve:
        if not np.isnan(x): v = x
        out.append(v)
    return m, np.array(out), tr

def daily(arr, step=24):
    return [r4(x) for x in arr[::step]] + ([r4(arr[-1])] if len(arr) % step != 1 else [])

def main():
    t0 = time.time()
    meta = json.load(open(f"{R}/sweep_meta.json")); sw = np.load(f"{R}/sweep.npz")
    IS, OOS, FULL = sw["is_"], sw["oos"], sw["full"]; combo, size, bal = sw["cfg_combo"], sw["cfg_size"], sw["cfg_bal"]
    combos = meta["combos"]; fam_of = np.array([E.F[c["family"]] for c in combos])[combo]
    D = prep(); chk = OV.check_arrays(D)
    T, split = D["T"], D["split"]
    metas = D["metas"]
    def strat_row(i, src="grid", extra=None):
        c = combos[combo[i]]; fr, mp = G.SIZING[size[i]]
        row = {"id": f"G{int(combo[i])}-S{int(size[i])}", "family": c["family"], "entry": entry_text(c["family"], c["p"]), "exit": exit_text(c["e"]),
               "sizing": sizing_text(fr, mp), "balance": float(bal[i]),
               "oos_end": r4(OOS[i, M["final"]]), "oos_profit": r4(OOS[i, M["final"]] - bal[i]), "oos_ret": r4(OOS[i, M["ret"]]),
               "oos_win": r4(OOS[i, M["winrate"]]), "oos_trades": int(OOS[i, M["ntrades"]]), "oos_dd": r4(OOS[i, M["maxdd"]]),
               "oos_sharpe": r4(OOS[i, M["sharpe"]]), "oos_avg_win": r4(OOS[i, M["avg_win"]]), "oos_avg_loss": r4(OOS[i, M["avg_loss"]]),
               "oos_pf": r4(OOS[i, M["pf"]]), "is_ret": r4(IS[i, M["ret"]]), "is_sharpe": r4(IS[i, M["sharpe"]]), "is_trades": int(IS[i, M["ntrades"]]),
               "is_win": r4(IS[i, M["winrate"]]), "is_dd": r4(IS[i, M["maxdd"]]), "full_ret": r4(FULL[i, M["ret"]]), "full_end": r4(FULL[i, M["final"]]),
               "full_win": r4(FULL[i, M["winrate"]]), "full_trades": int(FULL[i, M["ntrades"]]), "full_dd": r4(FULL[i, M["maxdd"]]), "src": src}
        w = WFD.get(i)
        if w:
            row.update(wf_share=r4(w["share_profitable"]), wf_active=w["active"], wf_median=r4(w["median_fold_ret"]), wf_robust=w["robust"])
        return row
    WFD = {}
    cand = (IS[:, M["ntrades"]] >= 10) & (OOS[:, M["ntrades"]] >= 10) & (IS[:, M["ret"]] > 0) & (IS[:, M["sharpe"]] > 0)
    def top_unique(score, mask, n=100, key=lambda i: (int(combo[i]), int(size[i]))):
        idx = np.where(mask)[0]; idx = idx[np.argsort(-score[idx])]
        out, seen = [], set()
        for i in idx:
            k = key(i)
            if k in seen: continue
            seen.add(k); out.append(int(i))
            if len(out) >= n: break
        return out
    top_sharpe = top_unique(OOS[:, M["sharpe"]], cand)
    top_profit = top_unique(OOS[:, M["ret"]], cand)
    # ---- walk-forward robustness on a broad candidate pool; the headline must pass it --------------------------
    pool = []
    for lst in (top_sharpe, top_profit[:100], top_unique(FULL[:, M["sharpe"]], cand, n=150)):
        pool += lst
    for fi in range(len(E.FAMILIES)):
        pool += top_unique(OOS[:, M["sharpe"]], cand & (fam_of == fi), n=15)
    pool = list(dict.fromkeys(pool))
    wf = {}
    for i in pool:
        cb = combos[combo[i]]; fr, mp = G.SIZING[size[i]]
        _, _, trs = resim(D, cb, fr, mp, float(bal[i]), 0, T)
        w = WF.folds(trs, D, float(bal[i]), fr, mp, k=26)
        w["robust"] = WF.is_robust(w, float(IS[i, M["ret"]]), float(OOS[i, M["ret"]]))
        wf[i] = w
    robust = sorted([i for i in pool if wf[i]["robust"]], key=lambda i: (-wf[i]["median_fold_ret"], -wf[i]["share_profitable"]))
    win = robust[0] if robust else (top_sharpe[0] if top_sharpe else int(np.argmax(OOS[:, M["sharpe"]])))
    print("walk-forward evaluated", len(pool), "robust", len(robust), flush=True)
    WFD.update(wf)
    is_pick = top_unique(IS[:, M["sharpe"]], (IS[:, M["ntrades"]] >= 20), n=1)[0]
    # per-balance leaderboard (top 25 by OOS sharpe)
    per_bal = {}
    for b in K.BALANCES:
        mb = cand & (bal == b)
        per_bal[f"{b:g}"] = [strat_row(i) for i in top_unique(OOS[:, M["sharpe"]], mb, n=25)]
    # best strategy return vs balance (winner's combo/size at every balance) + best-any per balance
    wc, ws = combo[win], size[win]
    vs_bal = []
    for b in K.BALANCES:
        i = np.where((combo == wc) & (size == ws) & (bal == b))[0][0]
        mb = cand & (bal == b)
        j = int(np.where(mb)[0][np.argmax(OOS[mb, M["ret"]])]) if mb.any() else i
        vs_bal.append({"balance": float(b), "winner_oos_ret": r4(OOS[i, M["ret"]]), "winner_trades": int(OOS[i, M["ntrades"]]),
                       "best_oos_ret": r4(OOS[j, M["ret"]]), "best_id": f"G{int(combo[j])}-S{int(size[j])}"})
    # equity curves at $100 (full period) for top 5 + benchmarks
    est_cols = [j for j in range(D["N"]) if D["grp"][j] == 0]
    all_cols = [j for j in range(D["N"]) if D["grp"][j] != 2]
    curves = {}
    wbal = float(bal[win])
    for rank, i in enumerate(top_sharpe[:5]):
        c = combos[combo[i]]; fr, mp = G.SIZING[size[i]]
        _, cv, _ = resim(D, c, fr, mp, 100.0, split, T)
        curves[f"#{rank+1} {c['family']}"] = daily(cv)
    sol = D["solpx"][split:T]; curves["SOL buy & hold"] = daily(100 * sol / sol[0] * (1 - K.PCT_COST[0]) ** 2)
    Cb = D["C"][split:T][:, est_cols]; ok = ~np.isnan(Cb[0])
    basket = np.nanmean(Cb[:, ok] / Cb[0, ok], axis=1) * 100 * (1 - K.PCT_COST[0]) ** 2
    curves["Meme basket buy & hold"] = daily(basket)
    # winner details: drawdown, per-coin
    c = combos[combo[win]]; fr, mp = G.SIZING[size[win]]
    mw, cvw, trw = resim(D, c, fr, mp, wbal, split, T)
    peak = np.maximum.accumulate(cvw); dd = cvw / peak - 1
    tok, ti, to, pin, pout, rs = trw
    selw = (ti >= split)
    per_coin = {}
    for k in np.where(selw)[0]:
        s = metas[tok[k]]["sym"]; g = pout[k] / pin[k] * (1 - K.PCT_COST[min(D["venue"][tok[k]], 2)]) ** 2 - 1
        d = per_coin.setdefault(s, {"symbol": s, "trades": 0, "wins": 0, "sum_ret": 0.0})
        d["trades"] += 1; d["wins"] += int(g > 0); d["sum_ret"] += g
    per_coin = sorted(per_coin.values(), key=lambda d: -d["sum_ret"])
    for d in per_coin: d["sum_ret"] = r4(d["sum_ret"])
    # in-sample equity for winner (for IS vs OOS chart)
    _, cv_is, _ = resim(D, c, fr, mp, wbal, 0, split)
    # benchmarks table
    bench = {}
    for name, lo, hi in (("is", 0, split), ("oos", split, T), ("full", 0, T)):
        s0, s1 = D["solpx"][lo], D["solpx"][hi - 1]
        bench[name] = {"sol": r4(s1 / s0 * (1 - K.PCT_COST[0]) ** 2 - 1), "basket_est": r4(bh(D, lo, hi, est_cols)[0]),
                       "basket_all": r4(bh(D, lo, hi, all_cols)[0]), "n_all": bh(D, lo, hi, all_cols)[1]}
    # heatmap: TP x SL median OOS return for std exits at $100 (all std-exit families)
    b100 = bal == 100.0
    tps = [0.02, 0.03, 0.05, 0.1, 0.25, 0.5, 1.0, E.NONE]; sls = [0.05, 0.1, 0.15, 0.2, 0.3, 0.35, 0.5]
    etp = np.array([c_["e"][0] for c_ in combos])[combo]; esl = np.array([c_["e"][1] for c_ in combos])[combo]
    hm = []
    for sl_ in sls:
        row = []
        for tp_ in tps:
            m_ = b100 & np.isclose(etp, tp_) & np.isclose(esl, sl_) & (OOS[:, M["ntrades"]] >= 5)
            row.append(r4(np.median(OOS[m_, M["ret"]])) if m_.sum() >= 5 else None)
        hm.append(row)
    heat = {"rows": [f"stop -{s*100:g}%" for s in sls], "cols": [("no TP" if t >= E.NONE / 2 else f"TP +{t*100:g}%") for t in tps], "values": hm}
    # distribution of OOS returns at $100
    r100 = OOS[b100 & (OOS[:, M["ntrades"]] >= 1), M["ret"]]
    edges = [-1, -0.9, -0.75, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 1, 2, 5, 1e9]
    hist = [{"label": (f"{edges[k]*100:g}% to {edges[k+1]*100:g}%" if edges[k + 1] < 1e8 else f"> {edges[k]*100:g}%"),
             "count": int(((r100 >= edges[k]) & (r100 < edges[k + 1])).sum())} for k in range(len(edges) - 1)]
    # family summary
    fam_sum = []
    for fi, fn in enumerate(E.FAMILIES):
        mf = b100 & (fam_of == fi)
        if not mf.any(): continue
        act = mf & (OOS[:, M["ntrades"]] >= 1)
        mc = mf & cand
        best = top_unique(OOS[:, M["sharpe"]], mc, n=1)
        fam_sum.append({"family": fn, "configs": int((fam_of == fi).sum()), "with_trades": int(act.sum()),
                        "median_oos_ret": r4(np.median(OOS[act, M["ret"]])) if act.any() else None,
                        "pct_profitable_oos": r4((OOS[act, M["ret"]] > 0).mean()) if act.any() else None,
                        "best": strat_row(best[0]) if best else None})
    # moonshot TP sweep
    ms = (fam_of == E.F["moonshot"]) & b100
    tp_rows = []
    for tp_ in sorted(set(np.round(etp[ms], 4))):
        m_ = ms & np.isclose(etp, tp_)
        if not m_.any(): continue
        tr_m = OOS[m_]; full_m = FULL[m_]
        if tp_ >= E.NONE / 2: continue   # runner (no TP) variants are reported in Max Return
        tp_rows.append({"tp_pct": r4(tp_ * 100), "configs": int(m_.sum()), "best_oos_ret": r4(tr_m[:, M["ret"]].max()),
                        "median_oos_ret": r4(np.median(tr_m[:, M["ret"]])), "best_full_ret": r4(full_m[:, M["ret"]].max()),
                        "median_full_ret": r4(np.median(full_m[:, M["ret"]]))})
    # TP hit rate from trades: fraction of moonshot trades exiting by take-profit, per TP level (full period, unique trades)
    trd = np.load(f"{R}/trades.npz"); offs = trd["offs"]
    for row in tp_rows:
        hits = 0; n = 0
        for ci, cb in enumerate(combos):
            if cb["family"] != "moonshot" or not np.isclose(cb["e"][0], row["tp_pct"] / 100) or cb["e"][1] != 1.0 or cb["e"][2] < E.NONE / 2 or cb["e"][6] < 1: continue
            a, b_ = offs[ci], offs[ci + 1]
            hits += int((trd["rs"][a:b_] == E.R_TP).sum()); n += int(b_ - a)
        row["hit_rate"] = r4(hits / n) if n else None; row["hit_n"] = n
    # launch outcome stats ("how often does 10x happen", "why they died")
    launch = [j for j in range(D["N"]) if D["grp"][j] == 1 and D["first"][j] >= 0]
    mult = []; died = 0; dead_list = []
    for j in launch:
        f0, l0 = D["first"][j], D["last"][j]
        p0 = D["C"][f0, j]
        if not p0 > 0: continue
        mx = np.nanmax(D["C"][f0 + 1:l0 + 1, j]) / p0 if l0 > f0 else 1.0   # close-based: ignores single-print wicks
        endr = D["C"][l0, j] / p0
        peak_ = np.nanmax(D["C"][f0:l0 + 1, j])
        mult.append(mx)
        if D["C"][l0, j] < 0.1 * peak_:
            died += 1
            v = D["V"][f0:l0 + 1, j]; hrs_to_peak = int(np.nanargmax(D["C"][f0:l0 + 1, j]))
            dead_list.append({"symbol": metas[j]["sym"], "peak_x": r4(peak_ / p0), "end_vs_peak": r4(D["C"][l0, j] / peak_ - 1),
                              "hours_to_peak": hrs_to_peak, "vol_last24_vs_peak24": r4(v[-24:].sum() / max(1.0, np.max(np.convolve(v, np.ones(24), "valid")))),
                              "top10_ex_lp": r4(D["top10"][j]) if not np.isnan(D["top10"][j]) else None, "mint_auth": bool(D["mint_auth"][j]),
                              "freeze_auth": bool(D["freeze"][j])})
    mult = np.array(mult)
    launch_stats = {"n": int(len(mult)), "hit_2x": r4((mult >= 2).mean()), "hit_5x": r4((mult >= 5).mean()), "hit_10x": r4((mult >= 10).mean()),
                    "hit_100x": r4((mult >= 100).mean()), "died_90pct_from_peak": died, "died_share": r4(died / max(1, len(mult)))}
    dead_list = sorted(dead_list, key=lambda d: -(d["peak_x"] or 0))[:12]
    # micro buy sizes: fixed $ per trade, real fee model; where do fees eat the trade?
    MICRO = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0, 2.0, 5.0, 10.0, 25.0]
    micro = []
    picks_m = (robust[:8] if robust else []) + [i for i in top_sharpe[:4] if i not in robust[:8]]
    sp_avg = float(np.mean(D["solpx"]))
    fixed_rt = (2 * K.FIXED_SOL + K.RENT_SOL * K.RENT_UNRECOVERED) * sp_avg
    for i in picks_m:
        cb = combos[combo[i]]
        _, _, trs = resim(D, cb, 0.1, 10, 100.0, 0, T)
        tok_, ti_, to_, pin_, pout_, _ = trs
        if len(tok_) == 0: continue
        ve_ = D["V"][np.maximum(ti_ - 1, 0), tok_]
        pc = K.PCT_COST[np.minimum(D["venue"][tok_], 2)]
        netpct = (pout_ / pin_) * (1 - pc) ** 2 - 1          # per-trade % after DEX fee + slippage, before fixed SOL costs
        mean_pct = float(np.mean(netpct))
        be = fixed_rt / mean_pct if mean_pct > 0 else None   # buy size where avg trade P&L = 0
        sims = []
        for s_ in MICRO:
            m_ = E.simulate(tok_, ti_, to_, pin_, pout_, D["venue"], ve_, D["solpx"], 0, T, 1000.0, -s_, 10, K.PCT_COST, K.FIXED_SOL,
                            K.RENT_SOL, K.RENT_UNRECOVERED, K.IMPACT_K, K.FILL_CAP, np.zeros(0))
            n_ = max(1.0, m_[4])
            sims.append({"size": s_, "trades": int(m_[4]), "pnl": r4(m_[0] - 1000.0), "pnl_per_trade": r4((m_[0] - 1000.0) / n_),
                         "fees_share_of_volume": r4(m_[11] / max(1e-9, s_ * n_ * 2))})
        micro.append({"id": strat_row(i)["id"], "family": cb["family"], "entry": entry_text(cb["family"], cb["p"]), "trades": int(len(tok_)),
                      "avg_trade_pct_after_pct_costs": r4(mean_pct), "fixed_cost_per_round_trip_usd": r4(fixed_rt), "breakeven_size_usd": r4(be), "sims": sims})
    # high win-rate section (OOS winrate >= 90%, >= 30 OOS trades)
    hw = np.where((OOS[:, M["winrate"]] >= 0.9) & (OOS[:, M["ntrades"]] >= 30))[0]
    win90 = [strat_row(i) for i in top_unique(OOS[:, M["ret"]], np.isin(np.arange(len(OOS)), hw), n=60)] if len(hw) else []
    closest = [] if len(hw) else [strat_row(i) for i in top_unique(OOS[:, M["winrate"]], OOS[:, M["ntrades"]] >= 30, n=15)]
    # max return (extreme upside)
    def topret(A, n=25, minn=1):
        return top_unique(A[:, M["ret"]], A[:, M["ntrades"]] >= minn, n=n)
    max_ret = {"oos": [strat_row(i) for i in topret(OOS)], "is": [strat_row(i) for i in topret(IS)], "full": [strat_row(i) for i in topret(FULL)]}
    extreme = {k: {"ret": r4(A[topret(A, 1)[0], M["ret"]]), "id": strat_row(topret(A, 1)[0])["id"]} for k, A in (("oos", OOS), ("is", IS), ("full", FULL))}
    extreme["reached_100000pct"] = bool(max(extreme[k]["ret"] for k in ("oos", "is", "full")) >= 1000)
    # best of year (full available window) by profit, candidates with >=10 trades
    best_year = [strat_row(i) for i in top_unique(FULL[:, M["ret"]], (FULL[:, M["ntrades"]] >= 10) & cand, n=100)]
    # desk exact post rules at $280
    desk_exact = None
    try:
        ci = next(k for k, cb in enumerate(combos) if cb["family"] == "desk" and int(cb["p"][4]) == 63 and np.isclose(cb["p"][0], 0.6)
                  and np.isclose(cb["p"][1], 0.2) and np.isclose(cb["p"][2], 5e4) and np.isclose(cb["e"][1], 0.06) and np.isclose(cb["e"][2], 0.12)
                  and np.isclose(cb["e"][3], 0.06) and cb["e"][4] == 24)
        out = {}
        for name, lo, hi in (("oos", split, T), ("full", 0, T)):
            m_, cv_, tr_ = resim(D, combos[ci], 0.08, 12, 280.0, lo, hi)
            out[name] = {"end": r4(m_[0]), "ret": r4(m_[1]), "trades": int(m_[4]), "wins": int(m_[5]), "losses": int(m_[4] - m_[5]),
                         "best": r4(m_[8]), "worst": r4(m_[9]), "maxdd": r4(m_[2]), "curve": daily(cv_)}
        # best 24h window of the full run (closest analogue to an "overnight" claim)
        cvf = np.array([x for x in out["full"]["curve"]], float)
        out["desk_id"] = ci
        desk_exact = out
    except StopIteration:
        pass
    # desk funnel: launch token-hours within 720h that pass each stage
    fun = {}
    lm = (D["grp"] == 1)
    age = np.arange(T)[:, None] - D["first"][None, :]
    base_m = (age >= 0) & (age <= 720) & lm[None, :] & ~np.isnan(D["C"])
    fun["scout_new_pairs_token_hours"] = int(base_m.sum())
    s1 = base_m & chk[1] & chk[2]; fun["after_mint_freeze"] = int(s1.sum())
    s2 = s1 & chk[3]; fun["after_top10_20pct"] = int(s2.sum())
    s3 = s2 & chk[4]; fun["after_volume_50k"] = int(s3.sum())
    s4 = s3 & chk[6]; fun["after_score_0.6"] = int(s4.sum())
    fun["rejected_share"] = r4(1 - s4.sum() / max(1, base_m.sum()))
    fun["tokens_scouted"] = int(lm.sum()); fun["tokens_passing_filters"] = int((s2.any(0)).sum())
    # question ablation from the desk_post grid (mask subsets)
    abl = []
    dm = [(k, cb) for k, cb in enumerate(combos) if cb["family"] == "desk" and np.isclose(cb["p"][2], 5e4) and np.isclose(cb["p"][1], 0.2) and np.isclose(cb["p"][0], 0.6) and len(cb["e"]) and np.isclose(cb["e"][1], 0.06) and np.isclose(cb["e"][2], 0.12)]
    if dm:
        sel_rows = {}
        for k, cb in dm:
            ii = np.where((combo == k) & (size == 0) & (bal == 280.0))[0]
            if len(ii): sel_rows[(int(cb["p"][4]), tuple(cb["e"]))] = OOS[ii[0]]
        names = ["mint", "freeze", "top10", "volume", "momentum", "score"]
        for b_ in range(6):
            on = [v[M["ret"]] for (mk, _), v in sel_rows.items() if mk & (1 << b_)]
            off = [v[M["ret"]] for (mk, _), v in sel_rows.items() if not mk & (1 << b_)]
            abl.append({"question": names[b_], "oos_ret_with": r4(np.mean(on)) if on else None, "oos_ret_without": r4(np.mean(off)) if off else None,
                        "edge": r4(np.mean(on) - np.mean(off)) if on and off else None, "trades_with": r4(np.mean([v[M["ntrades"]] for (mk, _), v in sel_rows.items() if mk & (1 << b_)])) if on else None})
    # deflated-sharpe style multiple-testing threshold
    uniq = np.where(b100 & (size == size[win]) & (OOS[:, M["ntrades"]] >= 10))[0]
    sr = OOS[uniq, M["sharpe"]]; Neff = len(uniq)
    g = 0.5772
    from statistics import NormalDist
    nd = NormalDist()
    exp_max = float(np.std(sr) * ((1 - g) * nd.inv_cdf(1 - 1 / Neff) + g * nd.inv_cdf(1 - 1 / (Neff * math.e)))) if Neff > 10 else None
    rep = {
        "generated": time.strftime("%Y-%m-%d %H:%M %Z"), "counts": {"grid_configs": meta["n_configs"], "grid_strategies": len(combos),
            "space_size": meta["space_size"], "trades_generated": meta["n_trades_total"]},
        "coverage": {"tokens": D["N"] - 1, "established": int((D["grp"] == 0).sum()), "launches": int((D["grp"] == 1).sum()),
                     "start": iso(meta["t0"]), "split": iso(meta["t_split"]), "end": iso(meta["t_end"]), "timeframe": "1h", "hours": T},
        "winner": strat_row(win), "is_pick": strat_row(is_pick),
        "walkforward": {"evaluated": len(pool), "robust": len(robust), "folds": 26, "rule": ">=8 active folds, >=60% of them profitable, median fold > 0, IS > 0 and OOS > 0",
                        "top": [dict(strat_row(i), fold_rets=wf[i]["fold_rets"]) for i in robust[:25]]}, "top100": [strat_row(i) for i in top_sharpe], "top100_profit": [strat_row(i) for i in top_profit],
        "per_balance": per_bal, "vs_balance": vs_bal, "curves": curves, "winner_curve": {"oos": daily(cvw), "is": daily(cv_is), "dd": daily(dd)},
        "per_coin": per_coin[:30], "bench": bench, "heat": heat, "hist": hist, "families": fam_sum, "moon_tp": tp_rows,
        "launch_stats": launch_stats, "dead": dead_list, "win90": win90, "win90_closest": closest, "max_ret": max_ret, "extreme": extreme,
        "best_year": best_year, "micro": micro, "desk_exact": desk_exact, "desk_funnel": fun, "desk_ablation": abl,
        "multiple_testing": {"n_eff": Neff, "expected_max_sharpe_null": r4(exp_max), "winner_oos_sharpe": r4(OOS[win, M["sharpe"]])},
    }
    # overlays
    if os.path.exists(f"{R}/overlay.npz"):
        ov = np.load(f"{R}/overlay.npz"); om = json.load(open(f"{R}/overlay_meta.json"))
        rows = om["rows"]; oc = ov["cfg_combo"]; ob = ov["cfg_bal"]; oO = ov["oos"]; oI = ov["is_"]
        sel = ob == 100.0
        agents = {"none": 0, "SCOUT": 1, "FILTER": 30, "ANALYST": 96, "REGIME": 128, "FILTER+ANALYST": 126, "SCOUT+FILTER": 31,
                  "SCOUT+FILTER+ANALYST": 127, "ALL CHECKS": 255}
        fams = sorted(set(r["family"] for r in rows))
        table = []
        for an, mk in agents.items():
            for risk in (0, 1):
                for ex in (0, 1):
                    label = an + (" +RISK" if risk else "") + (" +EXEC" if ex else "")
                    vals = []
                    for fn in fams:
                        ids = [k for k, r in enumerate(rows) if r["family"] == fn and r["mask"] == mk and r["risk"] == risk and r["executor"] == ex]
                        mm = sel & np.isin(oc, ids) & (oO[:, M["ntrades"]] >= 5)
                        vals.append(r4(np.median(oO[mm, M["ret"]])) if mm.any() else None)
                    table.append({"agents": label, "values": vals})
        # by number of checks
        bycnt = []
        for nck in range(9):
            vals = []
            for fn in fams:
                ids = [k for k, r in enumerate(rows) if r["family"] == fn and bin(r["mask"]).count("1") == nck and r["risk"] == 0 and r["executor"] == 0]
                mm = sel & np.isin(oc, ids) & (oO[:, M["ntrades"]] >= 5)
                vals.append(r4(np.median(oO[mm, M["ret"]])) if mm.any() else None)
            bycnt.append({"checks": nck, "values": vals})
        # marginal edge per check (OOS, $100)
        edge = []
        for b_ in range(8):
            on = sel & np.isin(oc, [k for k, r in enumerate(rows) if r["mask"] & (1 << b_)]) & (oO[:, M["ntrades"]] >= 5)
            off = sel & np.isin(oc, [k for k, r in enumerate(rows) if not r["mask"] & (1 << b_)]) & (oO[:, M["ntrades"]] >= 5)
            edge.append({"check": OV.CHECKS[b_], "median_with": r4(np.median(oO[on, M["ret"]])) if on.any() else None,
                         "median_without": r4(np.median(oO[off, M["ret"]])) if off.any() else None})
        for x in edge:
            x["edge"] = r4(x["median_with"] - x["median_without"]) if x["median_with"] is not None and x["median_without"] is not None else None
        best_ov = []
        for fn in fams:
            ids = [k for k, r in enumerate(rows) if r["family"] == fn]
            mm = sel & np.isin(oc, ids) & (oO[:, M["ntrades"]] >= 10) & (oI[:, M["ret"]] > 0)
            if mm.any():
                i = np.where(mm)[0][np.argmax(oO[mm, M["sharpe"]])]; r = rows[oc[i]]
                best_ov.append({"family": fn, "checks": OV.mask_text(r["mask"]), "risk": r["risk"], "executor": r["executor"],
                                "oos_ret": r4(oO[i, M["ret"]]), "oos_sharpe": r4(oO[i, M["sharpe"]]), "oos_trades": int(oO[i, M["ntrades"]]),
                                "is_ret": r4(oI[i, M["ret"]])})
        rep["overlay"] = {"configs": om["n_configs"], "families": fams, "agents_table": table, "by_checks": bycnt, "edge": edge, "best": best_ov}
    if os.path.exists(f"{R}/smart_search.json"):
        ss = json.load(open(f"{R}/smart_search.json"))
        rs_ = ss["results"]; best = {}
        for obj in ("sharpe", "winrate", "max_return"):
            rr = [r for r in rs_ if r["objective"] == obj and r["is"][4] >= (20 if obj != "max_return" else 5)]
            key = {"sharpe": lambda r: r["is"][3], "winrate": lambda r: r["is"][13] + (0 if r["is"][6] > r["is"][7] else -1), "max_return": lambda r: r["is"][1]}[obj]
            rr.sort(key=key, reverse=True)
            best[obj] = [{"family": r["family"], "entry": entry_text(r["family"], r["p"]), "exit": exit_text(r["e"]), "sizing": sizing_text(r["frac"], r["maxpos"]),
                          "checks": OV.mask_text(r["ovmask"]), "risk": r["risk"], "is_ret": r4(r["is"][1]), "is_sharpe": r4(r["is"][3]), "is_win": r4(r["is"][13]),
                          "is_trades": int(r["is"][4]), "oos_ret": r4(r["oos"][1]), "oos_sharpe": r4(r["oos"][3]), "oos_win": r4(r["oos"][13]),
                          "oos_trades": int(r["oos"][4]), "oos_dd": r4(r["oos"][2])} for r in rr[:10]]
        rep["smart"] = {"evaluated": ss["n_evaluated"], "best": best}
        for r in rs_:  # include smart-search hits in the 90%-win and extreme lists honestly
            o = r["oos"]
            if o[13] >= 0.9 and o[4] >= 30:
                rep["win90"].append({"id": "smart", "family": r["family"], "entry": entry_text(r["family"], r["p"]), "exit": exit_text(r["e"]),
                                     "sizing": sizing_text(r["frac"], r["maxpos"]), "balance": 100.0, "oos_ret": r4(o[1]), "oos_win": r4(o[13]),
                                     "oos_trades": int(o[4]), "oos_dd": r4(o[2]), "oos_avg_win": r4(o[14]), "oos_avg_loss": r4(o[15]),
                                     "oos_profit": r4(o[0] - 100), "oos_end": r4(o[0]), "oos_sharpe": r4(o[3]), "src": "smart"})
        rep["extreme"]["smart_is_max"] = r4(max(r["is"][1] for r in rs_)); rep["extreme"]["smart_oos_max"] = r4(max(r["oos"][1] for r in rs_))
    if rep["win90"]:
        rep["win90_closest"] = []
    # merge the separate strategy-hunter's leaderboard (read-only; /workspace/solstrats_hunt)
    hp = os.environ.get("SOLSTRATS_HUNT", "/workspace/solstrats_hunt/leaderboard.csv")
    for r in rep["top100"] + rep["top100_profit"]:
        r["robust"] = bool((r.get("is_ret") or 0) > 0 and (r.get("oos_ret") or 0) > 0)
    if os.path.exists(hp):
        hunt = []
        with open(hp) as f:
            for h in csv.DictReader(f):
                try:
                    row = {k: h.get(k) for k in ("id", "family", "entry", "exit", "sizing")}
                    for k in ("balance", "oos_end", "oos_profit", "oos_ret", "oos_win", "oos_dd", "oos_sharpe", "oos_pf", "oos_avg_win", "oos_avg_loss",
                              "is_ret", "is_sharpe", "is_win", "is_dd", "full_end", "full_ret", "full_win", "full_dd"):
                        row[k] = r4(float(h[k])) if h.get(k) not in (None, "") else None
                    for k in ("oos_trades", "is_trades", "full_trades"):
                        row[k] = int(float(h[k])) if h.get(k) not in (None, "") else 0
                    row["src"] = "hunter batch " + str(h.get("batch"))
                    row["robust"] = bool((row["is_ret"] or 0) > 0 and (row["oos_ret"] or 0) > 0)
                    hunt.append(row)
                except (ValueError, KeyError, TypeError):
                    continue
        hc = [h for h in hunt if h["is_trades"] >= 10 and h["oos_trades"] >= 10 and (h["is_ret"] or 0) > 0 and (h["is_sharpe"] or 0) > 0]
        rep["hunter"] = {"rows": len(hunt), "robust": sum(h["robust"] for h in hunt), "candidates": len(hc),
                         "top": sorted([h for h in hunt if h["robust"]], key=lambda h: -(h["oos_profit"] or 0))[:25]}
        rep["top100"] = sorted(rep["top100"] + hc, key=lambda r: -(r.get("oos_sharpe") or -99))[:100]
        rep["top100_profit"] = sorted(rep["top100_profit"] + hc, key=lambda r: -(r.get("oos_ret") or -99))[:100]
    rp = os.environ.get("SOLSTRATS_HUNT_ROBUST", "/workspace/solstrats_hunt/robust_leaderboard.csv")
    if os.path.exists(rp):
        try:
            with open(rp) as f:
                rep["hunter_robust"] = list(csv.DictReader(f))[:50]
        except (OSError, csv.Error):
            pass
    # 365-day daily context (CoinGecko)
    yr = []
    for fn in sorted(os.listdir("data/daily")) if os.path.isdir("data/daily") else []:
        try:
            d = json.load(open(f"data/daily/{fn}")); p = d["prices"]
            if len(p) < 30: continue
            yr.append({"symbol": d["sym"], "days": len(p), "start": iso(p[0][0] / 1000), "ret": r4(p[-1][1] / p[0][1] - 1),
                       "max_dd": r4(min(np.array([x[1] for x in p]) / np.maximum.accumulate([x[1] for x in p])) - 1)})
        except (OSError, ValueError, KeyError, ZeroDivisionError):
            continue
    rep["year_daily"] = sorted(yr, key=lambda r: -(r["ret"] or -9))
    if os.path.exists(f"{R}/year.json"):
        rep["year"] = json.load(open(f"{R}/year.json"))
    json.dump(rep, open(f"{R}/report.json", "w"))
    # CSVs
    def wcsv(fn, rows_):
        if not rows_: return
        keys = list(rows_[0].keys())
        with open(f"{R}/{fn}", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows_)
    wcsv("top100.csv", rep["top100"]); wcsv("top100_by_profit.csv", rep["top100_profit"]); wcsv("best_of_year.csv", rep["best_year"])
    wcsv("win90.csv", rep["win90"] or rep["win90_closest"]); wcsv("max_return_oos.csv", rep["max_ret"]["oos"]); wcsv("families.csv", [dict(f, best=(f["best"] or {}).get("id")) for f in fam_sum])
    if rep.get("year"): wcsv("best_of_year_12m_daily.csv", rep["year"]["top"])
    wcsv("leaderboard_per_balance.csv", [dict(r, view_balance=k) for k, v in per_bal.items() for r in v]); wcsv("moonshot_tp.csv", tp_rows)
    print("report built", round(time.time() - t0), "s")

if __name__ == "__main__":
    main()
