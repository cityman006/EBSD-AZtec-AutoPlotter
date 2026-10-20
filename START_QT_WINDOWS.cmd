@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
if not exist ".venv\Scripts\python.exe" (
  echo Run INSTALL_QT_WINDOWS.cmd first. Python 3.12 x64 is required.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -u gui_qt.py
if errorlevel 1 pause
