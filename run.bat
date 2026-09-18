@echo off
REM ============================================================
REM  run.bat - Setup and run Face & Emotion Detection System
REM  Usage examples (run from the project root folder):
REM    run.bat webcam
REM    run.bat video data\samples\your_video.mp4
REM    run.bat image data\samples
REM ============================================================

setlocal

REM --- Check the py launcher and Python 3.11 are available ---
where py >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python launcher 'py' not found. Install Python 3.11 from python.org and try again.
    pause
    exit /b 1
)
py -3.11 -c "" >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python 3.11 not found via 'py -3.11'. Install Python 3.11 from python.org and try again.
    pause
    exit /b 1
)

REM --- Create virtual environment if it doesn't exist ---
if not exist venv (
    echo Creating virtual environment with Python 3.11...
    py -3.11 -m venv venv
)

REM --- Activate virtual environment ---
call venv\Scripts\activate.bat

REM --- Install/upgrade core packaging tools first (fixes missing pkg_resources) ---
echo Upgrading pip, setuptools, wheel...
python -m pip install --upgrade pip setuptools wheel

REM --- Install dependencies ---
echo Installing dependencies...
pip install -r requirements.txt

REM --- Parse arguments: mode and optional input path ---
set MODE=%1
set INPUT=%2

if "%MODE%"=="" (
    echo No mode given, defaulting to: image --input data\samples
    set MODE=image
    set INPUT=data\samples
)

REM --- Run the project ---
if "%MODE%"=="webcam" (
    python src\main.py --mode webcam
) else (
    if "%INPUT%"=="" (
        echo ERROR: --input path required for mode "%MODE%".
        echo Example: run.bat video data\samples\your_video.mp4
        pause
        exit /b 1
    )
    python src\main.py --mode %MODE% --input %INPUT%
)

echo.
echo Done. Check the outputs\ folder for annotated frames, CSV log, and chart.
pause
