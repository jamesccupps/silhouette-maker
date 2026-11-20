@echo off
title Silhouette Maker
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  call setup.bat
  if errorlevel 1 ( pause & exit /b 1 )
)
rem A newer version of the app may need a library this setup does not have yet.
".venv\Scripts\python.exe" -c "import shapely, fontTools, flask" >nul 2>&1
if errorlevel 1 (
  call setup.bat
  if errorlevel 1 ( pause & exit /b 1 )
)
".venv\Scripts\python.exe" app.py
if errorlevel 1 pause
