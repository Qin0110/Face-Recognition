@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Face Recognition Attendance System - Setup

set "PY_VERSION=3.12.10"
set "PY_INSTALLER=%TEMP%\python-%PY_VERSION%-amd64.exe"
set "PY_URL=https://www.python.org/ftp/python/%PY_VERSION%/python-%PY_VERSION%-amd64.exe"
set "ENV_DIR=%~dp0env"
set "ENV_PY=%~dp0env\Scripts\python.exe"
set "SETUP_MARKER=%~dp0env\setup_complete.txt"

echo ============================================================
echo  Face Recognition Attendance System - Environment Setup
echo ============================================================
echo  An internet connection is required. The first setup can
echo  take 15-40 minutes depending on the connection speed.
echo ============================================================
echo.

echo [Step 1/4] Looking for Python 3.12 64-bit...
call :find_python
if defined BASE_PY goto python_ready

echo Python 3.12 was not found. Downloading Python %PY_VERSION%...
call :download "%PY_URL%" "%PY_INSTALLER%"
if errorlevel 1 goto fail_download

echo Installing Python %PY_VERSION% for the current user. Please wait...
start "" /wait "%PY_INSTALLER%" /quiet InstallAllUsers=0 PrependPath=0 Include_launcher=1 InstallLauncherAllUsers=0 Include_tcltk=1 Include_pip=1 Include_test=0
del /q "%PY_INSTALLER%" >nul 2>&1
call :find_python
if not defined BASE_PY goto fail_python_install

:python_ready
echo Using Python: "%BASE_PY%"
echo.

echo [Step 2/4] Preparing the virtual environment...
if exist "%SETUP_MARKER%" del /q "%SETUP_MARKER%"
if exist "%ENV_PY%" (
    "%ENV_PY%" -c "import sys, tkinter; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>&1
    if errorlevel 1 (
        echo The existing virtual environment cannot be used on this computer. Recreating it...
        rmdir /s /q "%ENV_DIR%"
    )
)
if not exist "%ENV_PY%" (
    "%BASE_PY%" -m venv "%ENV_DIR%"
    if errorlevel 1 goto fail_venv
)
echo.

echo [Step 3/4] Installing packages. This is the longest step...
"%ENV_PY%" -m pip install --upgrade pip --disable-pip-version-check
"%ENV_PY%" -m pip install --default-timeout=120 --retries 5 --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto fail_packages
echo.

echo [Step 4/4] Preparing the database, recognition model and trained data...
"%ENV_PY%" main.py --setup
if errorlevel 1 goto fail_prepare

> "%SETUP_MARKER%" echo Setup completed on %DATE% %TIME%

echo.
echo ============================================================
echo  Setup completed successfully.
echo  Double-click run.bat to start the application.
echo ============================================================
echo.
pause
exit /b 0

:fail_download
echo.
echo [ERROR] Python could not be downloaded. Check the internet connection and run setup.bat again.
goto end_fail

:fail_python_install
echo.
echo [ERROR] Python %PY_VERSION% could not be installed automatically.
echo Install Python 3.12 64-bit manually from https://www.python.org/downloads/ and run setup.bat again.
goto end_fail

:fail_venv
echo.
echo [ERROR] The virtual environment could not be created.
goto end_fail

:fail_packages
echo.
echo [ERROR] Package installation failed. Check the internet connection and run setup.bat again.
goto end_fail

:fail_prepare
echo.
echo [ERROR] The database, model or trained data could not be prepared. See the messages above.
goto end_fail

:end_fail
echo.
pause
exit /b 1

:find_python
set "BASE_PY="
call :try_python "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if defined BASE_PY exit /b 0
call :try_python "%ProgramFiles%\Python312\python.exe"
if defined BASE_PY exit /b 0
set "LAUNCHER_PY="
for /f "delims=" %%I in ('py -3.12 -c "import sys; print(sys.executable)" 2^>nul') do set "LAUNCHER_PY=%%I"
if defined LAUNCHER_PY call :try_python "%LAUNCHER_PY%"
exit /b 0

:try_python
if not exist "%~1" exit /b 1
"%~1" -c "import sys, tkinter, venv; sys.exit(0 if sys.version_info[:2] == (3, 12) and sys.maxsize > 2**32 else 1)" >nul 2>&1
if errorlevel 1 exit /b 1
set "BASE_PY=%~1"
exit /b 0

:download
where curl.exe >nul 2>&1
if errorlevel 1 goto download_powershell
curl.exe -L --fail --retry 3 -o "%~2" "%~1"
if not errorlevel 1 exit /b 0
:download_powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference = 'SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%~1' -OutFile '%~2'"
exit /b %errorlevel%
