"""Compact data for the Page (dedupes strategy text into a dictionary)."""
import json
r = json.load(open("results/report.json"))
S = {}; 
def sid(x):
    k = (x["family"], x["entry"], x["exit"], x["sizing"])
    if k not in S: S[k] = len(S)
    return S[k]
def row(x, kind="oos"):
    return [sid(x), x["balance"], x.get("oos_end"), x.get("oos_ret"), x.get("oos_win"), x.get("oos_trades"), x.get("oos_dd"),
            x.get("oos_sharpe"), x.get("is_ret"), 1 if x.get("robust", (x.get("is_ret") or 0) > 0 and (x.get("oos_ret") or 0) > 0) else 0,
            0 if x.get("src", "grid") == "grid" else 1]
R2 = lambda v: None if v is None else round(v, 3)
out = {
 "t100": [row(x) for x in r["top100"]], "tp100": [row(x) for x in r["top100_profit"]],
 "bal": {k: [row(x) for x in v[:3]] for k, v in r["per_balance"].items()},
 "vsb": [[v["balance"], R2(v["winner_oos_ret"]), R2(v["best_oos_ret"])] for v in r["vs_balance"]],
 "curves": {k: [None if x is None else round(x, 1) for x in v[::2]] for k, v in r["curves"].items()},
 "dd": [round(x * 100, 1) for x in r["winner_curve"]["dd"][::2]],
 "pc": [[c["symbol"][:8], c["trades"], c["wins"], R2(c["sum_ret"])] for c in r["per_coin"][:15]],
 "heat": r["heat"], "hist": [[h["label"], h["count"]] for h in r["hist"]],
 "fam": [[f["family"], f["configs"], R2(f["median_oos_ret"]), R2(f["pct_profitable_oos"]), R2((f["best"] or {}).get("oos_ret"))] for f in r["families"]],
 "tp": [[t["tp_pct"], R2(t["best_oos_ret"]), R2(t["median_oos_ret"]), R2(t["hit_rate"]), R2(t["best_full_ret"])] for t in r["moon_tp"]],
 "w90c": [row(x) + [R2(x.get("oos_avg_win")), R2(x.get("oos_avg_loss"))] for x in (r["win90"] or r["win90_closest"])[:10]],
 "maxret": {k: [row(x) + [R2(x.get("full_ret")), x.get("full_trades"), x.get("is_trades")] for x in v[:6]] for k, v in r["max_ret"].items()},
 "year": [[sid({"family": y["family"], "entry": y["entry"], "exit": y["exit"], "sizing": y["sizing"]}), y["balance"], y["end"], y["ret"], y["win"], y["trades"], y["dd"], y["is_ret"], y["oos_ret"]] for y in r["year"]["top"][:12]],
 "ov": {"fams": r["overlay"]["families"], "rows": [[a["agents"]] + [R2(v) for v in a["values"]] for a in r["overlay"]["agents_table"]],
        "cnt": [[c["checks"]] + [R2(v) for v in c["values"]] for c in r["overlay"]["by_checks"]], "edge": [[e["check"], R2(e["median_with"]), R2(e["median_without"]), R2(e["edge"])] for e in r["overlay"]["edge"]],
        "best": [[b["family"], b["checks"], b["risk"], b["executor"], R2(b["oos_ret"]), R2(b["oos_sharpe"]), b["oos_trades"], R2(b["is_ret"])] for b in r["overlay"]["best"]]},
 "dead": [[d["symbol"][:10], d["peak_x"], d["end_vs_peak"], d["hours_to_peak"], d["vol_last24_vs_peak24"], d["top10_ex_lp"]] for d in r["dead"][:10]],
 "hunt": [[sid(h), h["balance"], h["oos_profit"], h["oos_ret"], h["is_ret"], h["oos_trades"], h["oos_win"], h["id"]] for h in r["hunter"]["top"][:10]],
}
def short(t):
    t = t.replace(" of balance per trade, up to ", " per trade, max ").replace(" open", "")
    return t
out["S"] = [[k[0], k[1], k[2], short(k[3])] for k in S]
s = json.dumps(out, separators=(",", ":"))
open("results/page_data.json", "w").write(s)
print(len(s) // 1024, "KB", len(S), "strategies")

# live snapshot summary
import os
if os.path.exists("data/live/snapshot.json"):
    sn = json.load(open("data/live/snapshot.json")); st = {s["id"]: s for s in json.load(open("data/live/strategies.json"))}
    books = []
    for k, b in sn["books"].items():
        s0 = st.get(b["strat"], {}); cl = b["closed"]
        books.append([b["strat"], s0.get("family"), b["balance0"], round(b["equity"], 2), round(b["equity"] / b["balance0"] - 1, 4), len(cl),
                      sum(1 for c in cl if c["pnl"] > 0), len(b["open"]), R2(s0.get("bt_full_ret")), R2(s0.get("bt_win")), R2(s0.get("bt_dd")), R2(s0.get("bt_oos_ret"))])
    logs = []
    for e in sn["log"][-8:][::-1]:
        logs.append([e["agent"], e["action"], (e.get("symbol") or "")[:12], e.get("book"), e.get("reason"), e.get("price"),
                     [[q["agent"], q["q"], q["a"], q.get("value")] for q in (e.get("qa") or [])]])
    out["live"] = {"ts": sn["ts"], "started": sn["started"], "polls": sn["polls"], "http": sn["http_stats"], "errors": sn["errors"],
                   "prov": {k: v for k, v in sn["provider_status"].items() if k != "failures"}, "fails": sn["provider_status"]["failures"],
                   "books": books, "movers": [[t["symbol"][:10] if t["symbol"] else "?", t["dex"], R2(t["mcap"]), R2(t["chg_h1"]), R2(t["vol_h1"]), R2(t["liq"]), R2(t.get("age_h"))] for t in sn["fast_movers"][:12]],
                   "xdex": [[x["symbol"], x["buy_on"], x["sell_on"], x["gap_pct"], x["net_pct"], x["expected_pct_after_mev"]] for x in sn["cross_dex"][:6]],
                   "log": logs, "sol": sn["sol_price"]}
    s = json.dumps(out, separators=(",", ":"))
    open("results/page_data.json", "w").write(s)
    print("with live", len(s) // 1024, "KB")
