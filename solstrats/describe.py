"""Plain-English descriptions of strategies."""
from .engine import NONE

def _pct(x):
    return f"{x*100:g}%"

def _usd(x):
    if x >= 1e6: return f"${x/1e6:g}M"
    if x >= 1e3: return f"${x/1e3:g}k"
    return f"${x:g}"

def entry_text(fam, p):
    if fam == "breakout":
        s = f"Buy when price closes {('more than ' + _pct(p[1]) + ' ') if p[1] else ''}above its {int(p[0])}h high"
        return s + (f" on {p[2]:g}x average volume" if p[2] else "")
    if fam == "ma_cross":
        return f"Buy when the {int(p[0])}h average crosses above the {int(p[1])}h average" + (f" on {p[2]:g}x volume" if p[2] else "")
    if fam == "rsi_revert":
        return f"Buy when {int(p[0])}h RSI drops below {p[1]:g}" + (" while price is above its 7-day average" if p[2] else "")
    if fam == "vol_spike":
        return f"Buy a green hour of +{_pct(p[1])} or more on {p[0]:g}x the 24h average volume"
    if fam == "dip_buy":
        return f"Buy after a {_pct(p[1])} drop from the {int(p[0])}h high" + (" once an hour closes green" if p[2] else "")
    if fam == "moonshot":
        mc = "" if p[1] >= NONE / 2 else f", market cap under {_usd(p[1])}"
        return f"Buy a new launch in its first {int(p[0])}h{mc}, hourly volume over {_usd(p[2])}, hour up {_pct(p[3])}+"
    if fam == "desk":
        m = int(p[4]); q = [f"new pair under {int(p[3])}h old"]
        if m & 1: q.append("mint revoked")
        if m & 2: q.append("freeze revoked")
        if m & 4: q.append(f"top-10 holders under {_pct(p[1])}")
        if m & 8: q.append(f"volume over {_usd(p[2])}/h")
        if m & 16: q.append("6h momentum rising")
        if m & 32: q.append(f"analyst score at least {p[0]:g}")
        return "Desk pipeline: " + ", ".join(q)
    if fam == "fast_mc":
        return (f"Buy when market cap is {_usd(p[0])}-{_usd(p[1])} and grew {_pct(p[2])}+ in {int(p[3])}h"
                + (f" on {p[4]:g}x volume" if p[4] else ""))
    if fam == "graduation":
        return f"Buy a freshly created pool within {int(p[0])}h if up {_pct(p[1])}+ from its first price" + (f", volume over {_usd(p[2])}/h" if p[2] else "")
    if fam == "bonding_curve":
        return f"Buy a pump.fun curve token when its market cap crosses {_usd(p[0])}" + (f" with {_usd(p[1])}+/h volume" if p[1] else "")
    if fam == "narrative":
        return f"Daily, buy tokens in the hottest theme (by {int(p[0])}h return) that are up {_pct(p[1])}+ themselves"
    if fam == "flash_wick":
        return (f"Buy the next open after a single-hour dump of {_pct(p[0])}+" + (f" on {p[1]:g}x volume" if p[1] else "")
                + (f", only if the prior 24h fell less than {_pct(1 - p[2])}" if p[2] else ""))
    return fam

def exit_text(e):
    tp, sl, act, w, tm, mct, cr = e
    parts = []
    if tp < NONE / 2: parts.append(f"take profit +{_pct(tp)}")
    if mct < NONE / 2: parts.append(f"sell at {_usd(mct)} market cap")
    if sl < 1: parts.append(f"stop -{_pct(sl)}")
    if act < NONE / 2: parts.append(f"trail {_pct(w)} after +{_pct(act)}")
    if cr < 1: parts.append(f"exit on a -{_pct(cr)} hourly crash")
    parts.append(f"exit after {int(tm)}h")
    return ", ".join(parts)

def sizing_text(frac, maxpos):
    return f"{_pct(frac)} of balance per trade, up to {maxpos} open"
