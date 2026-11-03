@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
  py -3 -m pip install -r requirements.txt
) else (
  python -m pip install -r requirements.txt
)
if errorlevel 1 (
  echo.
  echo Setup failed. Install Python 3.10 or newer from https://python.org and try again.
  pause
  exit /b 1
)
echo.
echo FLIM Studio is ready. Run run_flim_studio.bat to start it.
pause
