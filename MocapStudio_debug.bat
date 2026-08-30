@echo off
rem Mocap Studio launcher with console (for error messages)
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m mocap_studio
) else (
  python -m mocap_studio
)
echo.
echo ---- exited (code %errorlevel%) ----
pause
