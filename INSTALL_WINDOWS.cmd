@echo off
setlocal
cd /d "%~dp0"
if errorlevel 1 exit /b 1
if not exist "%~dp0logs" mkdir "%~dp0logs"
if errorlevel 1 exit /b 1
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
if defined PYTHON312 goto custom
if exist "%LocalAppData%\Programs\Python\Python312\python.exe" (
  "%LocalAppData%\Programs\Python\Python312\python.exe" "%~dp0windows_setup.py" install %* 2>> "%~dp0logs\bootstrap_errors.log"
  goto done
)
where py >nul 2>nul
if errorlevel 1 goto missing
py -3.12 "%~dp0windows_setup.py" install %* 2>> "%~dp0logs\bootstrap_errors.log"
goto done
:custom
"%PYTHON312%" "%~dp0windows_setup.py" install %* 2>> "%~dp0logs\bootstrap_errors.log"
goto done
:missing
echo Python 3.12 x64 is required. Install from python.org or set PYTHON312.
echo Python 3.12 x64 is required. Install from python.org or set PYTHON312.>> "%~dp0logs\bootstrap_errors.log"
exit /b 1
:done
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" echo [%date% %time%] INSTALL_WINDOWS.cmd exit %RC%.>> "%~dp0logs\bootstrap_errors.log"
if not "%RC%"=="0" echo FAILED. See the logs folder.
if /i "%~1"=="--unattended" exit /b %RC%
pause
exit /b %RC%
