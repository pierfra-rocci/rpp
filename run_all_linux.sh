#!/bin/bash

set -euo pipefail

VENV_PATH=".venv/bin/activate"

if [ ! -f "$VENV_PATH" ]; then
    echo "Error: virtual environment not found at $VENV_PATH"
    exit 1
fi

source "$VENV_PATH"

if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "Error: failed to activate the virtual environment."
    exit 1
fi

export APP_ENV="${APP_ENV:-production}"
export RPP_API_URL="${RPP_API_URL:-http://127.0.0.1:8000}"
export RPP_LEGACY_URL="${RPP_LEGACY_URL:-http://127.0.0.1:5000}"

API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"
STREAMLIT_HOST="${STREAMLIT_HOST:-0.0.0.0}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"
GUNICORN_WORKERS="${GUNICORN_WORKERS:-2}"
GUNICORN_TIMEOUT="${GUNICORN_TIMEOUT:-60}"

backend_log="fastapi_backend.log"
frontend_log="streamlit_frontend.log"
backend_pid_file="fastapi_backend.pid"
frontend_pid_file="streamlit_frontend.pid"

cleanup() {
    echo
    echo "Shutting down services..."

    if [ -f "$backend_pid_file" ]; then
        backend_pid=$(cat "$backend_pid_file")
        if ps -p "$backend_pid" > /dev/null 2>&1; then
            echo "Stopping FastAPI backend (PID: $backend_pid)..."
            kill "$backend_pid" 2>/dev/null || true
            sleep 2
            kill -9 "$backend_pid" 2>/dev/null || true
        fi
        rm -f "$backend_pid_file"
    fi

    if [ -f "$frontend_pid_file" ]; then
        frontend_pid=$(cat "$frontend_pid_file")
        if ps -p "$frontend_pid" > /dev/null 2>&1; then
            echo "Stopping Streamlit frontend (PID: $frontend_pid)..."
            kill "$frontend_pid" 2>/dev/null || true
            sleep 2
            kill -9 "$frontend_pid" 2>/dev/null || true
        fi
        rm -f "$frontend_pid_file"
    fi

    pkill -f "gunicorn .* api.main:app" 2>/dev/null || true
    pkill -f "uvicorn api.main:app" 2>/dev/null || true
    pkill -f "streamlit run frontend.py" 2>/dev/null || true

    echo "Cleanup complete"
}

trap cleanup EXIT INT TERM

echo "Starting FastAPI backend with gunicorn..."
gunicorn \
    --workers "$GUNICORN_WORKERS" \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind "$API_HOST:$API_PORT" \
    --log-level info \
    --timeout "$GUNICORN_TIMEOUT" \
    api.main:app \
    >"$backend_log" 2>&1 &
backend_pid=$!
echo "$backend_pid" > "$backend_pid_file"
echo "FastAPI backend started with PID: $backend_pid"

echo "Waiting for backend to accept connections..."
sleep 5

if ! ps -p "$backend_pid" > /dev/null 2>&1; then
    echo "ERROR: FastAPI backend failed to start. Check $backend_log"
    exit 1
fi

echo "Starting Streamlit app..."
streamlit run frontend.py \
    --server.address "$STREAMLIT_HOST" \
    --server.port "$STREAMLIT_PORT" \
    >"$frontend_log" 2>&1 &
frontend_pid=$!
echo "$frontend_pid" > "$frontend_pid_file"
echo "Streamlit frontend started with PID: $frontend_pid"

sleep 3

if ! ps -p "$frontend_pid" > /dev/null 2>&1; then
    echo "ERROR: Streamlit frontend failed to start. Check $frontend_log"
    exit 1
fi

echo
    echo "Services started successfully."
    echo "FastAPI backend: http://$API_HOST:$API_PORT (PID: $backend_pid)"
    echo "Streamlit app:   http://$STREAMLIT_HOST:$STREAMLIT_PORT (PID: $frontend_pid)"
    echo "Backend log: tail -f $backend_log"
    echo "Frontend log: tail -f $frontend_log"
    echo
    echo "Press Ctrl+C to stop both services"

wait
