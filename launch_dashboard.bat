@echo off
echo.
echo ======================================================================
echo   LAUNCHING FINTECH AI PLATFORM DASHBOARD
echo ======================================================================
echo.
echo Starting Streamlit dashboard...
echo Dashboard will open in your browser automatically.
echo.
echo Press Ctrl+C to stop the dashboard
echo.
cd /d "%~dp0"
streamlit run dashboard.py
