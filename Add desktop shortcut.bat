@echo off
rem Puts a "Silhouette Maker" icon on the Desktop that starts the app.
setlocal
set "SM_DIR=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$d = $env:SM_DIR; $w = New-Object -ComObject WScript.Shell;" ^
  "$s = $w.CreateShortcut((Join-Path ([Environment]::GetFolderPath('Desktop')) 'Silhouette Maker.lnk'));" ^
  "$s.TargetPath = Join-Path $d 'Start Silhouette Maker.bat'; $s.WorkingDirectory = $d;" ^
  "$s.IconLocation = (Join-Path $d 'static\icon.ico'); $s.WindowStyle = 7;" ^
  "$s.Description = 'Turn side-on photos into silhouette cut files'; $s.Save()"
if errorlevel 1 ( echo The shortcut could not be made. & pause & exit /b 1 )
echo A "Silhouette Maker" shortcut is now on your Desktop.
