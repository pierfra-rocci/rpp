#!/bin/bash
# Activate virtual environment
source venv/bin/activate

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: Virtual environment not activated. Please run 'source venv/bin/activate' first."
    exit 1
fi

backend_log="backend.log"
frontend_log="frontend.log"

echo "Starting backend_dev.py in a new terminal..."
gnome-terminal -- bash -c "source venv/bin/activate; python backend_dev.py | tee $backend_log; exec bash"

echo "Starting frontend.py in a new terminal..."
gnome-terminal -- bash -c "source venv/bin/activate; streamlit run frontend.py | tee $frontend_log; exec bash"

echo "Backend URL: http://127.0.0.1:5000"
echo "Frontend URL: http://127.0.0.1:8501"
echo "Logs: $backend_log, $frontend_log"
echo "Both backend and frontend are running in separate terminals."
