"""Parameter-plateau check for washout-gate headline strategies, read from the hunter's batch 7 neighbourhood sweep (read-only).
For each centre (entry dn<N>_<SMA>, gate dump_<pct>_<hours>) it collects every batch-7 config within +-1 bar, +-24h SMA,
+-2 pct and +-4h (any exit/sizing tried there) and reports how many are profitable over the full span and the median return.
A wide profitable plateau is less likely to be a lucky single point. -> results/plateau.json"""
import csv, json, re, statistics as st
HUNT = "/workspace/solstrats_hunt"
CENTRES = {"H6-01098": (8, 120, 16, 16), "H7-02417": (8, 132, 17, 14), "H6-01718": (8, 288, 16, 24), "H6-00468": None}

def key(params):
    p = json.loads(params); m = re.match(r"dn(\d+)_(\d+)", p.get("entry", "")); g = re.match(r"dump_(\d+)_(\d+)", p.get("gate", ""))
    return (int(m[1]), int(m[2]), int(g[1]), int(g[2])) if m and g else None

def main(R="results"):
    rows = [r for r in csv.DictReader(open(f"{HUNT}/batch_7.csv"))]
    rob = {r["id"]: r for r in csv.DictReader(open(f"{HUNT}/robust_leaderboard.csv"))}
    for i in list(CENTRES):
        if CENTRES[i] is None and i in rob: CENTRES[i] = key(rob[i]["params"])
    allr = [float(r["full_ret"]) for r in rows]
    out = {"batch7_all": {"n": len(allr), "profitable": round(sum(x > 0 for x in allr) / len(allr), 4), "median_full_ret": round(st.median(allr), 4)}, "centres": []}
    for cid, c in CENTRES.items():
        if not c: continue
        nb = [r for r in rows if (k := key(r["params"])) and abs(k[0] - c[0]) <= 1 and abs(k[1] - c[1]) <= 24 and abs(k[2] - c[2]) <= 2 and abs(k[3] - c[3]) <= 4]
        fr = [float(r["full_ret"]) for r in nb]
        if not fr: continue
        out["centres"].append({"id": cid, "centre": "dn%d_%d dump_%d_%d" % c, "neighbours": len(fr), "distinct_entries": len({key(r["params"]) for r in nb}),
                               "profitable": round(sum(x > 0 for x in fr) / len(fr), 4), "median_full_ret": round(st.median(fr), 4),
                               "p10_full_ret": round(sorted(fr)[len(fr) // 10], 4), "oos_profitable": round(sum(float(r["oos_ret"]) > 0 for r in nb) / len(nb), 4)})
    json.dump(out, open(f"{R}/plateau.json", "w"), indent=1); print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
