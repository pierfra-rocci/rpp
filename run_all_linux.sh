#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: Virtual environment not activated. Please run 'source .venv/bin/activate' first."
    exit 1
fi

backend_log="backend.log"
frontend_log="frontend.log"
backend_pid_file="backend.pid"
frontend_pid_file="frontend.pid"
export APP_ENV=production

# -----------------------------------------------------------------------------
# IMPORTANT: Set the SMTP password for production before running the script.
# This should be set as an environment variable, not hardcoded here.
# For example, you can run: export SMTP_PASS="your_secret_password"
# -----------------------------------------------------------------------------
if [ -z "$SMTP_PASS" ]; then
    echo "Warning: SMTP_PASS environment variable is not set."
    echo "The application may not be able to send emails."
    echo ""
fi

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down services..."
    
    if [ -f "$backend_pid_file" ]; then
        backend_pid=$(cat "$backend_pid_file")
        if ps -p $backend_pid > /dev/null 2>&1; then
            echo "Stopping backend (PID: $backend_pid)..."
            kill $backend_pid 2>/dev/null
            # Wait a bit, then force kill if needed
            sleep 2
            kill -9 $backend_pid 2>/dev/null
        fi
        rm -f "$backend_pid_file"
    fi
    
    if [ -f "$frontend_pid_file" ]; then
        frontend_pid=$(cat "$frontend_pid_file")
        if ps -p $frontend_pid > /dev/null 2>&1; then
            echo "Stopping frontend (PID: $frontend_pid)..."
            kill $frontend_pid 2>/dev/null
            sleep 2
            kill -9 $frontend_pid 2>/dev/null
        fi
        rm -f "$frontend_pid_file"
    fi
    
    # Extra safety: kill any remaining processes
    pkill -f "gunicorn.*backend:app" 2>/dev/null
    pkill -f "streamlit run frontend.py" 2>/dev/null
    
    echo "Cleanup complete"
    exit 0
}

# Set up trap to call cleanup on script exit
trap cleanup EXIT INT TERM

echo "Starting backend with gunicorn in background..."
gunicorn --workers 2 --threads 4 --worker-class gthread --bind 0.0.0.0:5000 --log-level=info --timeout 60 --keep-alive 5 --error-logfile - backend:app > $backend_log 2>&1 &
backend_pid=$!
echo $backend_pid > $backend_pid_file
echo "Backend started with PID: $backend_pid"

echo "Waiting for backend to start..."
sleep 5

# Verify backend is running
if ! ps -p $backend_pid > /dev/null 2>&1; then
    echo "ERROR: Backend failed to start. Check $backend_log"
    exit 1
fi

echo "Starting frontend.py in background..."
streamlit run frontend.py > $frontend_log 2>&1 &
frontend_pid=$!
echo $frontend_pid > $frontend_pid_file
echo "Frontend started with PID: $frontend_pid"

sleep 3

# Verify frontend is running
if ! ps -p $frontend_pid > /dev/null 2>&1; then
    echo "ERROR: Frontend failed to start. Check $frontend_log"
    exit 1
fi

echo ""
echo "✓ Services started successfully!"
echo "Backend URL: http://127.0.0.1:5000 (PID: $backend_pid)"
echo "Frontend URL: http://127.0.0.1:8501 (PID: $frontend_pid)"
echo "Backend log: tail -f $backend_log"
echo "Frontend log: tail -f $frontend_log"
echo ""
echo "Press Ctrl+C to stop both services"

# Wait for both processes
wait