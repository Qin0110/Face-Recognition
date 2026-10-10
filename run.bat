@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Face Recognition Attendance System

set "ENV_PY=%~dp0env\Scripts\python.exe"
set "ENV_PYW=%~dp0env\Scripts\pythonw.exe"
set "SETUP_MARKER=%~dp0env\setup_complete.txt"

call :check_environment
if errorlevel 1 goto not_ready

:launch
start "" "%ENV_PYW%" "%~dp0main.py"
exit /b 0

:not_ready
echo The environment is not ready on this computer yet.
echo setup.bat needs to run once to install everything.
echo.
choice /C YN /M "Run setup.bat now"
if errorlevel 2 exit /b 1
call "%~dp0setup.bat"
call :check_environment
if errorlevel 1 exit /b 1
goto launch

:check_environment
if not exist "%SETUP_MARKER%" exit /b 1
if not exist "%ENV_PYW%" exit /b 1
"%ENV_PY%" -c "import sys" >nul 2>&1
if errorlevel 1 exit /b 1
exit /b 0
