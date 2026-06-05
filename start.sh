#!/bin/bash
set -e

echo ""
echo "============================================"
echo " AquaIntel -- Water Scarcity ML System"
echo "============================================"
echo ""

# Activate venv
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "[OK] Virtual environment activated"
fi

# Install backend deps
pip show fastapi > /dev/null 2>&1 || pip install -r backend_requirements.txt

# Generate dataset
if [ ! -f "data/processed/ganges_basin.npz" ]; then
    echo "[..] Generating synthetic dataset..."
    python data/download_datasets.py --synthetic
fi

# Start backend
echo "[..] Starting FastAPI backend on port 8000..."
uvicorn backend.server:app --reload --port 8000 --host 0.0.0.0 &
BACKEND_PID=$!
echo "[OK] Backend PID: $BACKEND_PID"

sleep 3

# Start frontend
cd frontend
[ ! -d "node_modules" ] && echo "[..] Installing npm packages..." && npm install
echo "[..] Starting React frontend on port 3000..."
npm start &
FRONTEND_PID=$!
cd ..

echo ""
echo "============================================"
echo " Backend API : http://localhost:8000"
echo " Frontend UI : http://localhost:3000"  
echo " API Docs    : http://localhost:8000/docs"
echo "============================================"
echo ""
echo "Press Ctrl+C to stop all services"

wait $BACKEND_PID $FRONTEND_PID
