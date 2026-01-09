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

# Redis configuration
REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}"
export CELERY_BROKER_URL="$REDIS_URL"
export CELERY_RESULT_BACKEND="$REDIS_URL"

API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"
STREAMLIT_HOST="${STREAMLIT_HOST:-0.0.0.0}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"
GUNICORN_WORKERS="${GUNICORN_WORKERS:-2}"
GUNICORN_TIMEOUT="${GUNICORN_TIMEOUT:-60}"
CELERY_WORKERS="${CELERY_WORKERS:-2}"
CELERY_CONCURRENCY="${CELERY_CONCURRENCY:-2}"

backend_log="fastapi_backend.log"
frontend_log="streamlit_frontend.log"
celery_log="celery_worker.log"
backend_pid_file="fastapi_backend.pid"
frontend_pid_file="streamlit_frontend.pid"
celery_pid_file="celery_worker.pid"

cleanup() {
    echo
    echo "Shutting down services..."

    # Stop Celery worker
    if [ -f "$celery_pid_file" ]; then
        celery_pid=$(cat "$celery_pid_file")
        if ps -p "$celery_pid" > /dev/null 2>&1; then
            echo "Stopping Celery worker (PID: $celery_pid)..."
            kill "$celery_pid" 2>/dev/null || true
            sleep 2
            kill -9 "$celery_pid" 2>/dev/null || true
        fi
        rm -f "$celery_pid_file"
    fi

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

    pkill -f "celery.*worker.*celery_app" 2>/dev/null || true
    pkill -f "gunicorn .* api.main:app" 2>/dev/null || true
    pkill -f "uvicorn api.main:app" 2>/dev/null || true
    pkill -f "streamlit run frontend.py" 2>/dev/null || true

    echo "Cleanup complete"
}

trap cleanup EXIT INT TERM

# Check Redis connectivity (optional - won't fail if Redis isn't running)
check_redis() {
    if command -v redis-cli &> /dev/null; then
        if redis-cli ping > /dev/null 2>&1; then
            echo "✓ Redis is available"
            return 0
        else
            echo "⚠ Redis is not responding. Background jobs will not work."
            echo "  Start Redis with: sudo systemctl start redis"
            return 1
        fi
    else
        echo "⚠ redis-cli not found. Cannot verify Redis status."
        return 1
    fi
}

echo "Checking Redis..."
redis_available=false
if check_redis; then
    redis_available=true
fi

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

# Start Celery worker if Redis is available
if [ "$redis_available" = true ]; then
    echo "Starting Celery worker..."
    celery -A celery_app worker \
        --loglevel=info \
        --concurrency="$CELERY_CONCURRENCY" \
        >"$celery_log" 2>&1 &
    celery_pid=$!
    echo "$celery_pid" > "$celery_pid_file"
    echo "Celery worker started with PID: $celery_pid"
    
    sleep 3
    
    if ! ps -p "$celery_pid" > /dev/null 2>&1; then
        echo "WARNING: Celery worker failed to start. Check $celery_log"
        echo "Background jobs will not be available."
    fi
else
    echo "Skipping Celery worker (Redis not available)"
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

echo ""
echo "Services started successfully."
echo "FastAPI backend: http://$API_HOST:$API_PORT (PID: $backend_pid)"
if [ "$redis_available" = true ] && [ -f "$celery_pid_file" ]; then
    echo "Celery worker:   Running (PID: $(cat $celery_pid_file))"
else
    echo "Celery worker:   Not running (Redis unavailable)"
fi
echo "Streamlit app:   http://$STREAMLIT_HOST:$STREAMLIT_PORT (PID: $frontend_pid)"
echo ""
echo "Logs:"
echo "  Backend:  tail -f $backend_log"
echo "  Celery:   tail -f $celery_log"
echo "  Frontend: tail -f $frontend_log"
echo ""
echo "Press Ctrl+C to stop all services"

wait
