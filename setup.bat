@echo off
rem One-time setup: makes a private Python environment in .venv and installs
rem what the app needs. Safe to run again.
setlocal
cd /d "%~dp0"

set "PY="
for %%V in (3.12 3.13 3.11 3.10) do if not defined PY (
  py -%%V -c "" >nul 2>&1 && set "PY=py -%%V"
)
if not defined PY (
  echo Python 3.10 to 3.13 is needed. Get it from https://www.python.org/downloads/
  echo ^(tick "Add python.exe to PATH" / install the py launcher^), then run this again.
  exit /b 1
)

echo Setting up with %PY% ...
if not exist ".venv\Scripts\python.exe" %PY% -m venv .venv || goto :fail
".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip || goto :fail
".venv\Scripts\python.exe" -m pip install --quiet -r requirements.txt || goto :fail

rem The best cut-out model needs the graphics card. On Windows that is the
rem DirectML build of onnxruntime, which replaces the plain one.
".venv\Scripts\python.exe" -m pip uninstall --quiet -y onnxruntime
".venv\Scripts\python.exe" -m pip install --quiet --force-reinstall --no-deps onnxruntime-directml || goto :fail

echo Setup finished.
exit /b 0

:fail
echo.
echo Setup did not finish. Check the message above, then run setup.bat again.
exit /b 1
