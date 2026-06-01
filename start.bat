@echo off
title AI Product Strategy Assistant
color 0A

echo ============================================================
echo   AI-Powered Product Strategy Assistant
echo ============================================================
echo.

:: Check for .env
if not exist .env (
    echo [WARNING] .env file not found.
    echo Please copy .env.example to .env and add your OPENAI_API_KEY
    echo.
    pause
    exit /b 1
)

echo [1/2] Starting FastAPI backend on http://localhost:8000 ...
start "Backend - FastAPI" cmd /k "cd /d %~dp0backend && python main.py"

timeout /t 3 /nobreak >nul

echo [2/2] Starting Streamlit frontend on http://localhost:8501 ...
start "Frontend - Streamlit" cmd /k "cd /d %~dp0frontend && streamlit run app.py --server.port 8501"

echo.
echo ============================================================
echo   Both servers are starting up!
echo   Backend API : http://localhost:8000
echo   Frontend UI : http://localhost:8501
echo   API Docs    : http://localhost:8000/docs
echo ============================================================
echo.
echo Press any key to exit this window (servers keep running) ...
pause >nul
