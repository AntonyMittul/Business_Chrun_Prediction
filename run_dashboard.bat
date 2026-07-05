@echo off
REM Launches the Streamlit dashboard with the project venv (NOT system Python).
REM Usage: double-click this file, or run .\run_dashboard.bat from the project root.
set VENV_PY=%USERPROFILE%\.venvs\churn\Scripts\python.exe
if not exist "%VENV_PY%" (
    echo Venv not found at %VENV_PY%
    echo Create it first:  python -m venv %USERPROFILE%\.venvs\churn ^&^& %USERPROFILE%\.venvs\churn\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)
cd /d "%~dp0"
"%VENV_PY%" -m streamlit run app\dashboard.py
