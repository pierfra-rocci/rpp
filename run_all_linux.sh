#!/bin/bash
# Activate virtual environment
source venv/bin/activate

# Start backend_dev.py in the background, redirecting output to backend.log
backend_log="backend.log"
frontend_log="frontend.log"
echo "Starting backend_dev.py..."
python backend_dev.py > "$backend_log" 2>&1 &
BACKEND_PID=$!

# Start run_frontend.py in the background, redirecting output to frontend.log

echo "Starting run_frontend.py..."
python run_frontend.py > "$frontend_log" 2>&1 &
FRONTEND_PID=$!

echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo "Backend URL: https://127.0.0.1:8000"
echo "Frontend URL: https://127.0.0.1:8501"
echo "Logs: $backend_log, $frontend_log"
echo "Both backend and frontend are running. Press Ctrl+C to stop."

# Wait for both to finish
wait $BACKEND_PID $FRONTEND_PID
