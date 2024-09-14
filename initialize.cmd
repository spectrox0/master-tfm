@echo off

REM Check if python is installed
where python >nul 2>nul
if %errorlevel% == 0 (
    echo Python is installed. Running initialize.py...
    python initialize.py
) else (
    echo Python is not installed. Please install Python to proceed.
    exit /b 1
)
