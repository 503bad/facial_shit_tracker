@echo off
rem Mocap Studio launcher (no console window)
cd /d "%~dp0"
where pythonw >nul 2>nul
if %errorlevel%==0 (
  start "" pythonw -m mocap_studio
) else (
  start "" python -m mocap_studio
)
