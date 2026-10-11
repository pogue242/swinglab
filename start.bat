@echo off
setlocal EnableExtensions
cd /d "%~dp0swinglab\swinglab"
if errorlevel 1 goto :fail

set "PY="
py -3 -c "import sys; raise SystemExit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1
if not errorlevel 1 (
  set "PY=py -3"
  goto :havepy
)
python -c "import sys; raise SystemExit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1
if not errorlevel 1 (
  set "PY=python"
  goto :havepy
)
echo Install Python 3.10 or newer from https://www.python.org/downloads/
echo Tick "Add python.exe to PATH" in the installer, then double-click this file again.
pause
exit /b 1

:havepy
if not exist .venv (
  echo Creating a Python environment...
  %PY% -m venv .venv
  if errorlevel 1 goto :fail
)
call .venv\Scripts\activate.bat
echo Installing SwingLab...
python -m pip install -e ".[phone,gemini,server]"
if errorlevel 1 goto :fail
if not exist "data\models\pose_landmarker_heavy.task" python scripts\get_pose_model.py
if errorlevel 1 goto :fail
if not exist "data\models\object_detection_yolox_2022nov.onnx" python scripts\get_ball_model.py
if errorlevel 1 goto :fail
if not exist .env copy /Y .env.example .env >nul
echo.
echo Opening http://localhost:8000
echo Coach reports use a template until you put a Gemini key in swinglab\swinglab\.env
echo Close this window to stop SwingLab.
start "" cmd /c "timeout /t 3 /nobreak >nul & start http://localhost:8000"
python api\server.py
if errorlevel 1 goto :fail
goto :eof

:fail
echo.
echo SwingLab did not start.
pause
exit /b 1
