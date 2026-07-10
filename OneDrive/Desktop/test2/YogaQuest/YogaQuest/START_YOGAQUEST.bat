@echo off
echo ========================================
echo   YogaQuest - AI Yoga Game Platform
echo ========================================
echo.

cd /d "%~dp0"

echo Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://python.org
    pause
    exit /b 1
)

echo Checking Streamlit...
python -m streamlit --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Streamlit is not installed
    echo Installing Streamlit...
    python -m pip install streamlit
)

echo.
echo Starting YogaQuest...
echo The app will open in your browser at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

python start_app.py

echo.
echo YogaQuest has stopped.
pause