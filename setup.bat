@echo off
rem One-time setup: makes a private Python environment in .venv and installs
rem what the app needs. Safe to run again.
rem   setup.bat            picks the graphics runtime for this PC
rem   setup.bat nvidia     NVIDIA's own runtime (NVIDIA cards; fastest)
rem   setup.bat directml   DirectML (any DirectX 12 graphics card)
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
set "VPY=.venv\Scripts\python.exe"
"%VPY%" -m pip install --quiet --upgrade pip || goto :fail
"%VPY%" -m pip install --quiet -r requirements.txt || goto :fail

rem The cut-out models run through onnxruntime, which comes in builds that
rem cannot be installed side by side. NVIDIA's own runtime makes the best
rem model about ten times faster than DirectML, but only NVIDIA cards have it.
rem Run without a choice, setup keeps a graphics runtime that already works.
set "RT=%~1"
if "%~1"=="" "%VPY%" -c "import engine, sys; sys.exit(engine.runtime() == 'cpu')" >nul 2>&1 && goto :done
if not defined RT set "RT=directml"
if "%~1"=="" nvidia-smi >nul 2>&1 && set "RT=nvidia"
"%VPY%" -c "import engine, sys; sys.exit(engine.runtime() != '%RT%')" >nul 2>&1 && goto :done
if /i not "%RT%"=="nvidia" goto :directml

echo.
echo This PC has an NVIDIA graphics card. Installing NVIDIA's runtime for the
echo best cut-out model: a download of about 2.2 GB, one time only.
"%VPY%" -m pip uninstall --quiet -y onnxruntime onnxruntime-directml onnxruntime-gpu
"%VPY%" -m pip install "onnxruntime-gpu[cuda,cudnn]" || goto :directml
"%VPY%" -c "import engine, sys; sys.exit(not engine.cuda_works())" && goto :done
echo.
echo NVIDIA's runtime could not run on this PC ^(the graphics driver may be too
echo old^). Using DirectML instead.

:directml
"%VPY%" -m pip uninstall --quiet -y onnxruntime onnxruntime-gpu onnxruntime-directml
for /f "delims==" %%P in ('%VPY% -m pip list --format^=freeze 2^>nul ^| findstr /b /i nvidia-') do "%VPY%" -m pip uninstall --quiet -y %%P
"%VPY%" -m pip install --quiet onnxruntime-directml || goto :fail

:done
echo Setup finished.
exit /b 0

:fail
echo.
echo Setup did not finish. Check the message above, then run setup.bat again.
exit /b 1
