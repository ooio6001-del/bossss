"""Package source, tests, docs, cached data and results into solstrats.zip (no keys, no HTTP caches, no big npz)."""
import glob, os, zipfile
FILES = ["README.md", "requirements.txt", ".env.example", ".gitignore", "run_all.sh", "make_zip.py", "dashboard.html",
         "run_backtest.py", "run_overlays.py", "search_smart.py", "run_year.py", "build_report.py", "build_html.py", "run_live.py",
         "fetch_history.py", "strategies.py", "compute_windows.py", "window_js.py", "serve.py", "micro_breakeven.py", "backtest_vs_live.py", "page_window.py", "page_compact.py", "stress.py", "null_test.py", "plateau.py", "monthly.py", "page_robust.py", "render_check.mjs", "fetch_year.py", "onchain.py", "discover.py", "discover2.py",
         "data/universe.json", "data/onchain.json", "data/live/snapshot.json", "data/live/strategies.json", "data/live/log.jsonl",
         "results/report.json", "results/year.json", "results/sweep_meta.json", "results/overlay_meta.json", "results/smart_search.json", "results/windows.json", "results/window_payload.json", "results/micro_breakeven.json", "results/backtest_vs_live.json", "results/page_window.json", "results/page_window_compact.json", "data/live/snapshot_run2.json", "data/live/snapshot_run3.json", "data/live/snapshot_run4.json", "data/live/snapshot_run5.json", "results/stress.json", "results/null_test.json", "results/plateau.json", "results/monthly.json", "results/page_robust.json"]
GLOBS = ["solstrats/*.py", "tests/*.py", "data/ohlcv/*.json", "data/daily/*.json", "results/*.csv"]
def main():
    keys = [v.strip() for v in (os.environ.get("HELIUS_API_KEY", ""), os.environ.get("BIRDEYE_API_KEY", "")) if v and v.strip()]
    paths = [f for f in FILES if os.path.exists(f)] + sorted(p for g in GLOBS for p in glob.glob(g))
    with zipfile.ZipFile("solstrats.zip.tmp", "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            data = open(p, "rb").read()
            for k in keys:
                if k.encode() in data:
                    raise SystemExit(f"key material found in {p}; refusing to package")
            z.writestr("solstrats/" + p, data)
    os.replace("solstrats.zip.tmp", "solstrats.zip")
    print("solstrats.zip", len(paths), "files", round(os.path.getsize("solstrats.zip") / 1e6, 1), "MB")
if __name__ == "__main__":
    main()
