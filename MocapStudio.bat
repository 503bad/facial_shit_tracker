@echo off
rem Mocap Studio launcher (no console window)
cd /d "%~dp0"
if not exist "%~dp0.venv\Scripts\pythonw.exe" goto no_venv
start "" "%~dp0.venv\Scripts\pythonw.exe" -m mocap_studio
exit /b 0

:no_venv
rem No local environment yet: use a system Python that already has the
rem dependencies, otherwise run the setup.
pythonw -c "import PySide6, mediapipe, cv2, pythonosc" >nul 2>nul
if errorlevel 1 goto run_setup
start "" pythonw -m mocap_studio
exit /b 0

:run_setup
call "%~dp0setup.bat"
