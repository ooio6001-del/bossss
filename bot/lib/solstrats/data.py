"""Load cached GeckoTerminal hourly OHLCV into aligned (T, N) arrays."""
from __future__ import annotations
import glob, json, os, re
import numpy as np

THEMES = [("ai", r"ai|gpt|agent|bot|swarm|griff|zere|arc|alch|neur|llm|16z"),
          ("dog", r"dog|inu|wif|shib|doge|bonk|pup|bark|chill"),
          ("cat", r"cat|mew|michi|kitty|meow|pop"),
          ("frog", r"frog|pepe|fwog|toad|ribbit"),
          ("political", r"trump|melania|maga|biden|president|libra|america|usa|elon|musk"),
          ("animal", r"goat|moo|deng|pnut|squirrel|hippo|ape|monkey|peng|tiger|rat|duck|fish|ponke|bear|bull")]

def theme_of(sym: str, name: str = "") -> int:
    s = f"{sym} {name}".lower()
    for i, (_, rx) in enumerate(THEMES):
        if re.search(rx, s):
            return i
    return len(THEMES)  # "other"

THEME_NAMES = [t for t, _ in THEMES] + ["other"]

def load(data_dir: str, max_days: int = 179, min_bars: int = 3):
    files = sorted(glob.glob(os.path.join(data_dir, "ohlcv", "*.json")))
    onchain = {}
    p = os.path.join(data_dir, "onchain.json")
    if os.path.exists(p):
        with open(p) as f:
            onchain = json.load(f)
    recs = []
    for fn in files:
        try:
            with open(fn) as f:
                d = json.load(f)
        except (OSError, ValueError):
            continue
        rows = d.get("ohlcv") or []
        if len(rows) < min_bars:
            continue
        recs.append((d["meta"], np.asarray(rows, dtype=np.float64)))
    if not recs:
        raise RuntimeError("no OHLCV data cached")
    t_end = max(r[1][:, 0].max() for r in recs)
    t0 = t_end - max_days * 86400
    T = int((t_end - t0) // 3600) + 1
    times = t0 + 3600 * np.arange(T)
    sol_idx = [i for i, (m, _) in enumerate(recs) if m.get("group") == "benchmark"]
    N = len(recs)
    O = np.full((T, N), np.nan); H = O.copy(); L = O.copy(); C = O.copy(); V = np.zeros((T, N))
    for j, (m, a) in enumerate(recs):
        a = a[a[:, 0] >= t0]
        idx = ((a[:, 0] - t0) // 3600).astype(int)
        ok = (idx >= 0) & (idx < T)
        idx, a = idx[ok], a[ok]
        O[idx, j], H[idx, j], L[idx, j], C[idx, j], V[idx, j] = a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5]
        # forward-fill gaps (hours with no trades) after first candle
        if len(idx):
            first = idx.min(); last = idx.max()
            for t in range(first + 1, last + 1):
                if np.isnan(C[t, j]):
                    C[t, j] = C[t - 1, j]; O[t, j] = C[t - 1, j]; H[t, j] = C[t - 1, j]; L[t, j] = C[t - 1, j]
    metas = [m for m, _ in recs]
    supply = np.zeros(N); mint_auth = np.zeros(N); freeze = np.zeros(N); top10 = np.full(N, np.nan)
    grp = np.zeros(N, dtype=np.int64); venue = np.zeros(N, dtype=np.int64); theme = np.zeros(N, dtype=np.int64)
    for j, m in enumerate(metas):
        oc = onchain.get(m["mint"], {})
        supply[j] = oc.get("supply") or 0.0
        if supply[j] <= 0 and m.get("mcap"):
            lastc = C[~np.isnan(C[:, j]), j]
            supply[j] = float(m["mcap"]) / lastc[-1] if len(lastc) and lastc[-1] > 0 else 1e9
        if supply[j] <= 0:
            supply[j] = 1e9
        mint_auth[j] = 1.0 if oc.get("mint_auth") else 0.0
        freeze[j] = 1.0 if oc.get("freeze_auth") else 0.0
        t10 = oc.get("top10_ex_lp", oc.get("top10_raw"))
        top10[j] = t10 if t10 is not None else np.nan
        grp[j] = {"established": 0, "launch": 1, "benchmark": 2}.get(m.get("group"), 1)
        dex = (m.get("dex") or "").lower()
        venue[j] = 2 if dex in ("pumpfun", "pump-fun") else (1 if grp[j] == 1 else 0)
        theme[j] = theme_of(m.get("sym") or "", m.get("name") or "")
    return dict(times=times, O=O, H=H, L=L, C=C, V=V, metas=metas, supply=supply, mint_auth=mint_auth,
                freeze=freeze, top10=top10, grp=grp, venue=venue, theme=theme, sol_idx=sol_idx[0] if sol_idx else -1)
