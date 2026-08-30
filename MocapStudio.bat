@echo off
rem Mocap Studio launcher (no console window)
cd /d "%~dp0"
if defined NV_AR_SDK_PATH if exist "%NV_AR_SDK_PATH%\nvARPose.dll" goto sdk_ok
if defined NVAR_MODEL_DIR if exist "%NVAR_MODEL_DIR%\..\nvARPose.dll" goto sdk_ok
if exist "%ProgramFiles%\NVIDIA Corporation\NVIDIA AR SDK\nvARPose.dll" goto sdk_ok
goto run_setup
:sdk_ok
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
