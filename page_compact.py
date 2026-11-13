"""Shrink results/page_window.json into the literal embedded in the Page (results/page_window_compact.json)."""
import json
d = json.load(open("results/page_window.json"))
B = d["bounds"]; T = B["48h"]["t_hi"]; base = T - d["span"]
order = ["1h", "4h", "12h", "24h", "48h", "7d", "30d", "90d", "12m"]
f4 = lambda x: float(f"{x:.4g}")
G, gi, S = [], {}, []
for s in d["strats"]:
    assert s["p"]["ytd"] == s["p"]["12m"]  # hourly data starts after Jan 1, so YTD == 12m == full span
    c = s["c"]
    if c:
        c = ";".join(",".join([j, ti, to, f"{float(g):.4f}", f"{float(pin):.5g}", f"{float(ve):.3g}", eo]) for j, ti, to, g, pin, ve, eo in (k.split(",") for k in c.split(";")))
        if c not in gi: gi[c] = len(G); G.append(c)
        gidx = gi[c]
    else:
        gidx = -2 if s["cx"] else -1   # -2: custom-capable but no candidate trades in the 48h span
    S.append([s["id"], s["tag"], s["fam"], s["rule"][:64], s["bal"], round(s["frac"], 4), s["mp"], s["one"], gidx,
              [[round(m[0], 2), m[1], round(m[2], 2), round(m[3], 2), m[4], round(m[5], 2)] for m in (s["p"][p] for p in order)]])
out = {"venue": d["venue"], "sol": [round(x, 1) for x in d["sol"]], "marks": [[f4(x) for x in m] for m in d["marks"]],
       "b": {p: [B[p]["t_lo"] - base, B[p]["hours"], B[p]["start"], B[p]["end"], int(B[p]["capped_to_data"])] for p in order},
       "g": G, "s": S}
txt = json.dumps(out, separators=(",", ":"), ensure_ascii=False)
open("results/page_window_compact.json", "w").write(txt); print(len(txt), "groups", len(G), "strategies", len(S))
