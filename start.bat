@echo off
title AquaIntel Launcher
color 0B

echo.
echo  ============================================
echo   AquaIntel -- Water Scarcity ML System
echo  ============================================
echo.

REM ── Activate venv ─────────────────────────────
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo [OK] Virtual environment activated
) else (
    echo [WARN] No venv found - using system Python
)

REM ── Install backend deps if needed ─────────────
echo [..] Checking backend dependencies...
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo [..] Installing FastAPI...
    pip install -r backend_requirements.txt
)

REM ── Generate dataset if not present ────────────
if not exist "data\processed\ganges_basin.npz" (
    echo [..] Generating synthetic dataset...
    python data\download_datasets.py --synthetic
)

REM ── Start FastAPI backend in new window ─────────
echo [..] Starting FastAPI backend on port 8000...
start "AquaIntel Backend" cmd /k "call venv\Scripts\activate.bat 2>nul & uvicorn backend.server:app --reload --port 8000 --host 0.0.0.0"

REM ── Wait for backend to be ready ───────────────
echo [..] Waiting for backend...
timeout /t 4 /nobreak >nul

REM ── Start React frontend in new window ──────────
echo [..] Starting React frontend on port 3000...
cd frontend
if not exist "node_modules" (
    echo [..] Installing npm packages (first run -- takes ~2 min)...
    npm install
)
start "AquaIntel Frontend" cmd /k "npm start"
cd ..

echo.
echo  ============================================
echo   Services starting:
echo   Backend API : http://localhost:8000
echo   Frontend UI : http://localhost:3000
echo   API Docs    : http://localhost:8000/docs
echo  ============================================
echo.
echo  Press any key to open the browser...
pause >nul
start http://localhost:3000
