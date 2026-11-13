"""Emit compact TS literals for the Page's robustness section (stress + null + monthly + plateau) -> results/page_robust.json"""
import json
def r2(x): return None if x is None else round(x * 100, 1)
st = json.load(open("results/stress.json")); nl = {r["id"]: r for r in json.load(open("results/null_test.json"))["rows"]}
mo = json.load(open("results/monthly.json")); pl = json.load(open("results/plateau.json"))
rows = st["rows"][:16]
STRESS = [[r["id"], r["balance"], r["trades"], r2(r["base"]), r2(r["cost_x2"]), r2(r["cost_x3"]), r2(r["late_1bar"]), r2(r["drop_best_token"]),
           r2(r["sub80_p05"]), round(nl[r["id"]]["p_value"], 4) if r["id"] in nl else -1, 1 if r["survives"] else 0] for r in rows]
MONTHLY = {"labels": mo["labels"], "rows": [[r["id"]] + [r2(x) for x in r["months"]] + [r2(r["total"])] for r in mo["rows"]]}
ns = [nl[r["id"]] for r in st["rows"] if r["id"] in nl]
SUM = {**st["summary"], "null_n": len(ns), "null_p05": sum(x["p_value"] <= 0.05 for x in ns), "null_draws": json.load(open("results/null_test.json"))["draws"],
       "plateau": next(c for c in pl["centres"] if c["id"] == "H6-01098")}
out = {"STRESS": STRESS, "MONTHLY": MONTHLY, "SUM": SUM}
json.dump(out, open("results/page_robust.json", "w"))
print(json.dumps(out)[:3000]); print(len(json.dumps(out)))
