#!/usr/bin/env bash
# SOLSTRATS paper trader: create venv, install pinned deps, start. PAPER ONLY - no wallet, no real trades.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
"$PY" -c 'import sys; assert sys.version_info >= (3,10), "Python 3.10+ required"'
if [ ! -d .venv ]; then "$PY" -m venv .venv; fi
. .venv/bin/activate
python -m pip install --upgrade pip >/dev/null
python -m pip install -r requirements.txt
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export SOLSTRATS_HOME="$(pwd)"
mkdir -p run
if [ "${1:-}" = "--foreground" ] || [ "${1:-}" = "--once" ]; then
  exec python live_paper.py "$@"
fi
nohup python live_paper.py >> run/trader.out 2>&1 &
echo "paper trader started, pid $! ; logs: run/events.log  run/trader.out ; dashboard: run/dashboard.html"
echo "stop it with: kill \$(cat run/trader.pid)"
