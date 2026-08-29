@echo off
rem Mocap Studio launcher with console (for error messages)
cd /d "%~dp0"
python -m mocap_studio
echo.
echo ---- exited (code %errorlevel%) ----
pause
