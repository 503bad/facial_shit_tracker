@echo off
setlocal
cd /d "%~dp0"
title Mocap Studio Setup
set "ROOT=%~dp0"
set "NVAR_MISSING="
set "PY="

echo ==========================================================
echo   Mocap Studio setup
echo   Checks Python / NVIDIA AR SDK, then installs the required
echo   libraries into a local ".venv" folder and starts the app.
echo   (Japanese guide: README.md)
echo ==========================================================
echo.

rem ---------- 1. NVIDIA AR SDK ----------
if exist "%ProgramFiles%\NVIDIA Corporation\NVIDIA AR SDK\nvARPose.dll" goto sdk_ok
echo [NG] NVIDIA AR SDK not found.
echo      Required for facial tracking (RTX GPU needed).
echo      Download the "AR SDK" redistributable and install it:
echo      https://www.nvidia.com/ja-jp/geforce/broadcasting/broadcast-sdk/resources/
echo.
start "" "https://www.nvidia.com/ja-jp/geforce/broadcasting/broadcast-sdk/resources/"
set "NVAR_MISSING=1"
goto sdk_done
:sdk_ok
echo [OK] NVIDIA AR SDK found.
:sdk_done
echo.

rem ---------- 2. Python 3.10 - 3.12 ----------
py -3.12 -c "import sys" >nul 2>nul && set "PY=py -3.12"
if defined PY goto py_found
py -3.11 -c "import sys" >nul 2>nul && set "PY=py -3.11"
if defined PY goto py_found
py -3.10 -c "import sys" >nul 2>nul && set "PY=py -3.10"
if defined PY goto py_found
python -c "import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>nul && set "PY=python"
if defined PY goto py_found

echo [NG] Python 3.10 - 3.12 not found.
echo      Install Python 3.12 from the page below
echo      (check "Add python.exe to PATH" in the installer):
echo      https://www.python.org/downloads/windows/
echo.
start "" "https://www.python.org/downloads/windows/"
echo After installing Python, run this setup.bat again.
pause
exit /b 1

:py_found
for /f "tokens=*" %%A in ('%PY% -c "import sys;print(sys.version.split()[0])"') do set "PYFOUND=%%A"
echo [OK] Using Python %PYFOUND% (%PY%)
echo.

rem ---------- 3. venv ----------
if exist "%ROOT%.venv\Scripts\python.exe" goto venv_ok
echo [..] Creating virtual environment .venv ...
%PY% -m venv "%ROOT%.venv"
if errorlevel 1 goto venv_fail
if not exist "%ROOT%.venv\Scripts\python.exe" goto venv_fail
:venv_ok
set "VPY=%ROOT%.venv\Scripts\python.exe"

rem ---------- 4. dependencies (re-run when requirements.txt changed) ----------
fc /b "%ROOT%requirements.txt" "%ROOT%.venv\requirements.installed" >nul 2>nul
if not errorlevel 1 goto deps_ok
echo [..] Installing libraries (first run takes a few minutes) ...
"%VPY%" -m pip install --upgrade pip >nul 2>nul
"%VPY%" -m pip install -r "%ROOT%requirements.txt"
if errorlevel 1 goto deps_fail
copy /y "%ROOT%requirements.txt" "%ROOT%.venv\requirements.installed" >nul
goto deps_done
:deps_ok
echo [OK] Libraries already installed.
:deps_done
echo.

rem ---------- 5. launch ----------
if defined NVAR_MISSING goto no_sdk
echo [OK] Setup complete. Starting Mocap Studio.
start "" "%ROOT%.venv\Scripts\pythonw.exe" -m mocap_studio
exit /b 0

:no_sdk
echo Install the NVIDIA AR SDK, then start with MocapStudio.bat.
pause
exit /b 0

:venv_fail
echo [NG] Failed to create the virtual environment.
pause
exit /b 1

:deps_fail
echo [NG] Failed to install the libraries.
pause
exit /b 1
