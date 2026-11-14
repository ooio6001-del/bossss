@echo off
REM SOLSTRATS paper trader: create venv, install pinned deps, start. PAPER ONLY - no wallet, no real trades.
cd /d "%~dp0"
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
%PY% -c "import sys; assert sys.version_info >= (3,10), 'Python 3.10+ required'" || exit /b 1
if not exist .venv %PY% -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul
python -m pip install -r requirements.txt || exit /b 1
if exist .env for /f "usebackq eol=# tokens=1,* delims==" %%a in (".env") do set "%%a=%%b"
set SOLSTRATS_HOME=%cd%
if not exist run mkdir run
echo Paper trader running in this window. Close it or press Ctrl+C to stop. Logs: run\events.log  Dashboard: run\dashboard.html
python live_paper.py %*
