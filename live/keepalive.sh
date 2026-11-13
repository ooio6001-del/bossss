#!/bin/bash
# restart the paper trader if it is not running (state.json makes restarts lossless)
cd /workspace/solstrats_live
PID=$(cat trader.pid 2>/dev/null)
if [ -z "$PID" ] || ! kill -0 "$PID" 2>/dev/null || ! grep -q live_paper.py /proc/$PID/cmdline 2>/dev/null; then
  nohup python3 live_paper.py >> trader.out 2>&1 &
  echo "$(date) restarted" >> keepalive.log
fi
