#!/bin/bash
# Activate virtual environment
source venv/bin/activate

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: Virtual environment not activated. Please run 'source venv/bin/activate' first."
    exit 1
fi

# Start backend_dev.py in the background, redirecting output to backend.log
backend_log="backend.log"
frontend_log="frontend.log"
echo "Starting backend_dev.py..."
python backend_dev.py > "$backend_log" 2>&1 &
BACKEND_PID=$!

# Start run_frontend.py in the background, redirecting output to frontend.log

echo "Starting frontend.py..."
streamlit run frontend.py > "$frontend_log" 2>&1 &
FRONTEND_PID=$!

echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo "Backend URL: http://127.0.0.1:5000"
echo "Frontend URL: http://127.0.0.1:8501"
echo "Logs: $backend_log, $frontend_log"
echo "Both backend and frontend are running. Press Ctrl+C to stop."

# Wait for both to finish
wait $BACKEND_PID $FRONTEND_PID
