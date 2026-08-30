@echo off
setlocal
cd /d "%~dp0"
title Mocap Studio - make release package
set "ROOT=%~dp0"

rem ---- version from mocap_studio/__init__.py ----
set "VER="
for /f "tokens=2 delims==" %%A in ('findstr /r "^__version__" "%ROOT%mocap_studio\__init__.py"') do set "VER=%%A"
set "VER=%VER: =%"
set "VER=%VER:"=%"
if "%VER%"=="" set "VER=dev"

set "NAME=MocapStudio-v%VER%"
set "OUT=%ROOT%release\%NAME%"
set "ZIP=%ROOT%release\%NAME%.zip"

echo ==========================================================
echo   Building release package: %NAME%
echo   -> %OUT%
echo ==========================================================
echo.

if exist "%OUT%" rmdir /s /q "%OUT%"
if exist "%ZIP%" del /q "%ZIP%"
mkdir "%OUT%" 2>nul

rem ---- application package (code + bundled MediaPipe models) ----
robocopy "%ROOT%mocap_studio" "%OUT%\mocap_studio" /E /NFL /NDL /NJH /NJS /NP ^
  /XD __pycache__ /XF *.pyc >nul
if errorlevel 8 goto copy_fail

rem ---- docs ----
robocopy "%ROOT%docs" "%OUT%\docs" /E /NFL /NDL /NJH /NJS /NP >nul
if errorlevel 8 goto copy_fail

rem ---- top-level files ----
for %%F in (setup.bat MocapStudio.bat MocapStudio_debug.bat requirements.txt README.md) do (
  copy /y "%ROOT%%%F" "%OUT%\%%F" >nul || goto copy_fail
)

rem ---- sanity: nothing private slipped in ----
if exist "%OUT%\settings.json" del /q "%OUT%\settings.json"
if exist "%OUT%\.venv" rmdir /s /q "%OUT%\.venv"

rem ---- zip ----
powershell -NoProfile -Command "Compress-Archive -Path '%OUT%' -DestinationPath '%ZIP%' -Force"
if errorlevel 1 goto zip_fail

echo Contents:
dir /s /b "%OUT%" | findstr /v "__pycache__"
echo.
for %%Z in ("%ZIP%") do echo [OK] %%~nxZ  (%%~zZ bytes)
echo.
echo Done. Distribute the zip (or the folder) in release\.
pause
exit /b 0

:copy_fail
echo [NG] copying files failed.
pause
exit /b 1

:zip_fail
echo [NG] creating the zip failed.
pause
exit /b 1
