"""Smart search over the huge nominal space (grid.SPACE): random sampling + Optuna TPE (Bayesian) per objective.
Fitness is computed IN-SAMPLE only; out-of-sample is recorded for honest validation, never optimized on."""
from __future__ import annotations
import json, math, os, sys, time, logging
import numpy as np
import optuna
from solstrats import engine as E, costs as K, grid as G, overlay as OV
from run_backtest import prep

optuna.logging.set_verbosity(optuna.logging.WARNING)
FAMS = list(G.SPACE.keys())

def sample(trial, fam):
    sp = G.SPACE[fam]; p = [0.0] * 6
    def f(name, lo, hi, step, log=False):
        return trial.suggest_float(f"{fam}.{name}", lo, hi, step=None if log else step, log=log)
    if fam == "breakout": p[:3] = [trial.suggest_int("bo.L", 2, 720, log=True), f("th", 0, 0.5, 0.0005), f("vm", 0, 20, 0.01)]
    elif fam == "ma_cross":
        fa = trial.suggest_int("ma.f", 2, 96, log=True); p[:3] = [fa, trial.suggest_int("ma.s", max(24, fa + 1), 720, log=True), f("vm", 0, 20, 0.01)]
    elif fam == "rsi_revert": p[:3] = [trial.suggest_int("rsi.per", 2, 72), f("thr", 2, 50, 0.01), trial.suggest_int("rsi.tr", 0, 1)]
    elif fam == "vol_spike": p[:2] = [f("vm", 1.5, 50, 0.01), f("r", 0, 1, 0.0005)]
    elif fam == "dip_buy": p[:3] = [trial.suggest_int("dip.L", 6, 720, log=True), f("d", 0.05, 0.95, 0.0005), trial.suggest_int("dip.b", 0, 1)]
    elif fam == "moonshot": p[:4] = [trial.suggest_int("ms.age", 1, 720, log=True), f("mc", 5e3, 5e7, 1e3, True), f("vol", 1, 1e6, 100, True), f("ret", -0.5, 3, 0.001)]
    elif fam == "desk": p[:5] = [f("score", 0.3, 0.95, 0.001), f("t10", 0.05, 0.9, 0.001), f("vol", 1, 1e6, 100, True), trial.suggest_int("desk.age", 6, 2160, log=True), trial.suggest_int("desk.mask", 0, 63)]
    elif fam == "fast_mc":
        lo = f("mcmin", 2500, 1e5, 100, True); p[:5] = [lo, max(lo * 1.5, f("mcmax", 1e4, 3e6, 1e3, True)), f("g", 0.1, 10, 0.001, True), trial.suggest_int("fmc.n", 1, 48, log=True), f("vm", 0, 20, 0.01)]
    elif fam == "graduation": p[:3] = [trial.suggest_int("gr.age", 1, 72, log=True), f("ret", -0.9, 5, 0.001), f("vol", 1, 1e6, 100, True)]
    elif fam == "bonding_curve": p[:2] = [f("mcx", 3e3, 6.9e4, 100, True), f("vol", 0, 1e5, 10)]
    elif fam == "narrative": p[:2] = [trial.suggest_categorical("nar.L", [24, 72, 168]), f("ret", -0.5, 2, 0.001)]
    elif fam == "flash_wick": p[:3] = [f("dump", 0.03, 0.9, 0.001), f("vm", 0, 20, 0.01), f("ff", 0, 1, 0.01)]
    tp = trial.suggest_float("x.tp", 0.005, 50, log=True) if trial.suggest_int("x.use_tp", 0, 1) else E.NONE
    sl = trial.suggest_float("x.sl", 0.01, 1.0, log=True)
    if trial.suggest_int("x.use_trail", 0, 1):
        act, w = trial.suggest_float("x.act", 0.01, 20, log=True), trial.suggest_float("x.w", 0.01, 0.9, log=True)
    else:
        act, w = E.NONE, 0.0
    tm = trial.suggest_int("x.tm", 1, 2160, log=True)
    mct = trial.suggest_float("x.mct", 1e4, 3e6, log=True) if fam in ("fast_mc", "bonding_curve", "moonshot") and trial.suggest_int("x.use_mct", 0, 1) else E.NONE
    cr = trial.suggest_float("x.crash", 0.05, 1.0)
    frac = trial.suggest_float("s.frac", 0.03, 1.0, log=True)
    maxpos = trial.suggest_int("s.maxpos", 1, 50, log=True)
    ovmask = trial.suggest_int("ov.mask", 0, 255) if trial.suggest_int("ov.use", 0, 1) else 0
    risk = trial.suggest_int("ov.risk", 0, 1)
    return p, [tp, sl, act, w, tm, mct, cr], frac, maxpos, ovmask, risk

def evaluate(D, chk, fam, p, e, frac, maxpos, ovmask, risk, balance=100.0):
    fi = E.F[fam]
    hot = D["hot"].get(int(p[0]), D["hot"][24]) if fam == "narrative" else D["hot"][24]
    sig = E.signals(fi, np.asarray(p, float), D["O"], D["H"], D["L"], D["C"], D["V"], D["first"], D["grp"], D["venue"],
                    D["supply"], D["mint_auth"], D["freeze"], D["top10"], hot, D["theme"], 0, D["T"] - 1)
    sig = OV.apply_checks(sig, chk, ovmask)
    if risk: e = list(OV.risk_exit(tuple(e)))
    one = fam in ("moonshot", "graduation", "bonding_curve")
    tr = E.gen_trades(sig, np.asarray(e, float), D["O"], D["H"], D["L"], D["C"], D["last"], D["supply"], one)
    ve = D["V"][np.maximum(tr[1] - 1, 0), tr[0]]
    out = {}
    for name, lo, hi in (("is", 0, D["split"]), ("oos", D["split"], D["T"])):
        out[name] = E.simulate(*tr[:5], D["venue"], ve, D["solpx"], lo, hi, balance, frac, maxpos, K.PCT_COST, K.FIXED_SOL,
                               K.RENT_SOL, K.RENT_UNRECOVERED, K.IMPACT_K, K.FILL_CAP, np.zeros(0))
    return out

OBJECTIVES = {
    "sharpe": lambda m: m[3] if m[4] >= 20 else -10 + m[4] / 10,
    "winrate": lambda m: (m[13] if (m[6] - m[7]) > 0 else m[13] - 1) if m[4] >= 30 else -1 + m[4] / 100,
    "max_return": lambda m: math.log1p(max(m[1], -0.999)) if m[4] >= 5 else -10,
}

def main(n_random=150, n_tpe=350, out="results/smart_search.json"):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    D = prep(); chk = OV.check_arrays(D)
    results = []; t0 = time.time()
    for obj, fit in OBJECTIVES.items():
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(n_startup_trials=n_random, seed=42))
        def objective(trial):
            fam = trial.suggest_categorical("family", FAMS)
            p, e, frac, maxpos, ovmask, risk = sample(trial, fam)
            try:
                r = evaluate(D, chk, fam, p, e, frac, maxpos, ovmask, risk)
            except Exception as ex:  # noqa: BLE001
                return -100.0
            results.append({"objective": obj, "family": fam, "p": p, "e": e, "frac": frac, "maxpos": maxpos, "ovmask": ovmask,
                            "risk": risk, "is": r["is"].tolist(), "oos": r["oos"].tolist()})
            return float(fit(r["is"]))
        study.optimize(objective, n_trials=n_random + n_tpe)
        print(obj, "best IS", round(study.best_value, 3), "elapsed", round(time.time() - t0), flush=True)
    json.dump({"n_evaluated": len(results), "space_size": G.space_size(), "results": results}, open(out, "w"))
    print("DONE", len(results))

if __name__ == "__main__":
    a = [int(x) for x in sys.argv[1:3]]
    main(*a)
