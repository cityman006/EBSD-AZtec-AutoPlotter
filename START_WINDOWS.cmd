@echo off
setlocal
cd /d "%~dp0"
if errorlevel 1 exit /b 1
if not exist "%~dp0logs" mkdir "%~dp0logs"
if errorlevel 1 exit /b 1
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo Project environment missing. Run REPAIR_WINDOWS.cmd.
  echo Project environment missing. Run REPAIR_WINDOWS.cmd.>> "%~dp0logs\bootstrap_errors.log"
  exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -u "%~dp0windows_setup.py" start %* 2>> "%~dp0logs\bootstrap_errors.log"
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" echo [%date% %time%] START_WINDOWS.cmd exit %RC%.>> "%~dp0logs\bootstrap_errors.log"
if "%RC%"=="0" exit /b 0
echo Launch failed. See the logs folder and run REPAIR_WINDOWS.cmd.
if /i "%~1"=="--unattended" exit /b %RC%
pause
exit /b %RC%
