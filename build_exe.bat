@echo off
setlocal
cd /d "%~dp0"

rem Prefer whichever interpreter actually has PyInstaller installed, rather
rem than guessing from "py" vs "python" alone -- on machines with more than
rem one Python install (e.g. Anaconda's "python" plus a separate "py"
rem launcher pointing elsewhere), those can resolve to different
rem interpreters, and installing PyInstaller into one doesn't help if this
rem script ends up using the other.

set PY=

python -m PyInstaller --version >nul 2>nul
if not errorlevel 1 (
    set PY=python
    goto :found
)

py -3 -m PyInstaller --version >nul 2>nul
if not errorlevel 1 (
    set PY=py -3
    goto :found
)

echo PyInstaller is not installed in any Python environment found on PATH.
echo Install project dependencies and PyInstaller with the same Python you
echo use to run FLIM Studio from source, e.g.:
echo     python -m pip install -r requirements.txt
echo     python -m pip install pyinstaller
echo (use "py -3 -m pip install ..." instead if that's the one you use)
pause
exit /b 1

:found
echo Using "%PY%" (has PyInstaller installed) to build.
echo Building FLIM Studio.exe with PyInstaller...
%PY% -m PyInstaller --noconfirm flim_studio.spec
if errorlevel 1 (
    echo.
    echo Build failed -- see the PyInstaller output above.
    pause
    exit /b 1
)

echo.
echo Build complete: dist\FLIM Studio\FLIM Studio.exe
echo Distribute the whole "dist\FLIM Studio" folder, not just the .exe.
pause
