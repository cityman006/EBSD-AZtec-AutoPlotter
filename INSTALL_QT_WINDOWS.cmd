@echo off
setlocal
cd /d "%~dp0"
call INSTALL_WINDOWS.cmd --unattended
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements-gui.txt
if errorlevel 1 goto failed
echo Qt GUI installed. Use START_QT_WINDOWS.cmd. Tk fallback: START_WINDOWS.cmd.
pause
exit /b 0
:failed
echo Installation failed. See logs and console output.
pause
exit /b 1
