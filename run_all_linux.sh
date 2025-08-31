#!/bin/bash
# Activate virtual environment
source .venv/bin/activate

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: Virtual environment not activated. Please run 'source venv/bin/activate' first."
    exit 1
fi

backend_log="backend.log"
frontend_log="frontend.log"

echo "Starting backend with gunicorn in background..."
gunicorn --workers 4 --bind 0.0.0.0:5000 --log-level=debug --error-logfile - backend:app 2>&1 | tee $backend_log &

echo "Waiting for backend to start..."
sleep 5

echo "Starting frontend.py in background..."
streamlit run frontend.py | tee $frontend_log &

echo "Backend URL: http://127.0.0.1:5000"
echo "Frontend URL: http://127.0.0.1:8501"
echo "Logs: $backend_log, $frontend_log"
echo "Use 'tail -f backend.log' or 'tail -f frontend.log' to view logs."
echo "Both backend and frontend are running in the background."
wait