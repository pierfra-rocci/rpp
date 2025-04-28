#!/bin/bash
# Activate virtual environment
source venv/bin/activate

# Start backend_dev.py in the background
echo "Starting backend_dev.py..."
python backend_dev.py &
BACKEND_PID=$!

# Start run_frontend.py in the background (or remove '&' to run in foreground)
echo "Starting run_frontend.py..."
python run_frontend.py &
FRONTEND_PID=$!

echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo "Both backend and frontend are running. Press Ctrl+C to stop."

# Wait for both to finish
wait $BACKEND_PID $FRONTEND_PID
