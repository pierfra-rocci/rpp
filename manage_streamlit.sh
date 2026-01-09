#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${SCRIPT_DIR}"
SCREEN_NAME="rpp_app"

# Warn if running as root (except for clean)
if [ "$EUID" -eq 0 ] && [ "${1:-}" != "clean" ]; then
    echo "Warning: Running as root is not recommended."
    echo "The screen session and processes should be owned by your regular user."
    echo "Consider running without sudo: ./manage_streamlit.sh $1"
    echo ""
fi

case "$1" in
    status)
        echo "=== Screen Sessions ==="
        screen -ls | grep "$SCREEN_NAME" || echo "No screen session found"
        echo ""
        echo "=== Running Processes ==="
        echo "FastAPI/Gunicorn:"
        pgrep -fa "gunicorn.*api.main:app" || echo "  Not running"
        echo "Celery Worker:"
        pgrep -fa "celery.*worker" || echo "  Not running"
        echo "Streamlit:"
        pgrep -fa "streamlit.*frontend" || echo "  Not running"
        echo ""
        echo "=== Redis Status ==="
        if command -v redis-cli &> /dev/null; then
            if redis-cli ping > /dev/null 2>&1; then
                echo "Redis: ✓ Running"
            else
                echo "Redis: ✗ Not responding"
            fi
        else
            echo "Redis: ? (redis-cli not installed)"
        fi
        echo ""
        if [ -f "$APP_DIR/fastapi_backend.pid" ]; then
            backend_pid=$(cat "$APP_DIR/fastapi_backend.pid" 2>/dev/null)
            if [ -n "$backend_pid" ] && ps -p "$backend_pid" > /dev/null 2>&1; then
                echo "Backend PID: $backend_pid (running)"
            else
                echo "Backend PID file exists but process not running"
            fi
        fi
        if [ -f "$APP_DIR/celery_worker.pid" ]; then
            celery_pid=$(cat "$APP_DIR/celery_worker.pid" 2>/dev/null)
            if [ -n "$celery_pid" ] && ps -p "$celery_pid" > /dev/null 2>&1; then
                echo "Celery PID: $celery_pid (running)"
            else
                echo "Celery PID file exists but process not running"
            fi
        fi
        if [ -f "$APP_DIR/streamlit_frontend.pid" ]; then
            frontend_pid=$(cat "$APP_DIR/streamlit_frontend.pid" 2>/dev/null)
            if [ -n "$frontend_pid" ] && ps -p "$frontend_pid" > /dev/null 2>&1; then
                echo "Frontend PID: $frontend_pid (running)"
            else
                echo "Frontend PID file exists but process not running"
            fi
        fi
        ;;
    attach)
        if screen -ls | grep -q "$SCREEN_NAME"; then
            screen -r "$SCREEN_NAME"
        else
            echo "No screen session named '$SCREEN_NAME' found."
            exit 1
        fi
        ;;
    restart)
        echo "Restarting application..."
        $0 stop
        sleep 3
        $0 start
        ;;
    stop)
        echo "Stopping application..."
        
        # Stop via screen session first (run_all_linux.sh cleanup trap will handle graceful shutdown)
        if screen -ls | grep -q "$SCREEN_NAME"; then
            echo "Sending quit command to screen session..."
            screen -S "$SCREEN_NAME" -X quit 2>/dev/null || true
            sleep 4
        fi
        
        # Stop Celery worker
        if [ -f "$APP_DIR/celery_worker.pid" ]; then
            celery_pid=$(cat "$APP_DIR/celery_worker.pid" 2>/dev/null)
            if [ -n "$celery_pid" ] && ps -p "$celery_pid" > /dev/null 2>&1; then
                echo "Stopping Celery worker $celery_pid..."
                kill "$celery_pid" 2>/dev/null || true
                sleep 2
                if ps -p "$celery_pid" > /dev/null 2>&1; then
                    kill -9 "$celery_pid" 2>/dev/null || true
                fi
            fi
            rm -f "$APP_DIR/celery_worker.pid"
        fi
        
        # Fallback: kill processes using PID files
        if [ -f "$APP_DIR/fastapi_backend.pid" ]; then
            backend_pid=$(cat "$APP_DIR/fastapi_backend.pid" 2>/dev/null)
            if [ -n "$backend_pid" ] && ps -p "$backend_pid" > /dev/null 2>&1; then
                echo "Stopping backend process $backend_pid..."
                kill "$backend_pid" 2>/dev/null || true
                sleep 2
                # Force kill if still running
                if ps -p "$backend_pid" > /dev/null 2>&1; then
                    kill -9 "$backend_pid" 2>/dev/null || true
                fi
            fi
            rm -f "$APP_DIR/fastapi_backend.pid"
        fi
        
        if [ -f "$APP_DIR/streamlit_frontend.pid" ]; then
            frontend_pid=$(cat "$APP_DIR/streamlit_frontend.pid" 2>/dev/null)
            if [ -n "$frontend_pid" ] && ps -p "$frontend_pid" > /dev/null 2>&1; then
                echo "Stopping frontend process $frontend_pid..."
                kill "$frontend_pid" 2>/dev/null || true
                sleep 2
                # Force kill if still running
                if ps -p "$frontend_pid" > /dev/null 2>&1; then
                    kill -9 "$frontend_pid" 2>/dev/null || true
                fi
            fi
            rm -f "$APP_DIR/streamlit_frontend.pid"
        fi
        
        # Final cleanup - check for any remaining processes
        pkill -f "celery.*worker.*celery_app" 2>/dev/null || true
        pkill -f "gunicorn.*api.main:app" 2>/dev/null || true
        pkill -f "streamlit.*frontend" 2>/dev/null || true
        
        echo "✓ Application stopped"
        ;;
    clean)
        echo "Cleaning up ALL streamlit, celery and backend processes..."
        pkill -9 -f "streamlit" || true
        pkill -9 -f "gunicorn" || true
        pkill -9 -f "uvicorn" || true
        pkill -9 -f "celery" || true
        screen -S "$SCREEN_NAME" -X quit 2>/dev/null || true
        rm -f "$APP_DIR/fastapi_backend.pid" "$APP_DIR/streamlit_frontend.pid" "$APP_DIR/celery_worker.pid"
        rm -f "$APP_DIR/fastapi_backend.log" "$APP_DIR/streamlit_frontend.log" "$APP_DIR/celery_worker.log"
        echo "✓ Cleanup complete"
        ;;
    logs)
        if [ -f "$APP_DIR/fastapi_backend.log" ] || [ -f "$APP_DIR/streamlit_frontend.log" ] || [ -f "$APP_DIR/celery_worker.log" ]; then
            case "${2:-}" in
                backend|1)
                    tail -f "$APP_DIR/fastapi_backend.log"
                    ;;
                celery|2)
                    tail -f "$APP_DIR/celery_worker.log"
                    ;;
                frontend|3)
                    tail -f "$APP_DIR/streamlit_frontend.log"
                    ;;
                all|4)
                    tail -f "$APP_DIR/fastapi_backend.log" "$APP_DIR/celery_worker.log" "$APP_DIR/streamlit_frontend.log"
                    ;;
                *)
                    echo "Available logs:"
                    echo "  $0 logs backend   - View backend logs (follow)"
                    echo "  $0 logs celery    - View Celery worker logs (follow)"
                    echo "  $0 logs frontend  - View frontend logs (follow)"
                    echo "  $0 logs all       - View all logs (follow)"
                    echo ""
                    echo "Recent backend log:"
                    tail -15 "$APP_DIR/fastapi_backend.log" 2>/dev/null || echo "  No backend log"
                    echo ""
                    echo "Recent Celery log:"
                    tail -15 "$APP_DIR/celery_worker.log" 2>/dev/null || echo "  No Celery log"
                    echo ""
                    echo "Recent frontend log:"
                    tail -15 "$APP_DIR/streamlit_frontend.log" 2>/dev/null || echo "  No frontend log"
                    ;;
            esac
        else
            echo "No log files found in $APP_DIR"
            if [ -f "$HOME/rpp_output.log" ]; then
                echo "Try: tail -f ~/rpp_output.log"
            fi
        fi
        ;;
    redis)
        case "${2:-}" in
            start)
                echo "Starting Redis..."
                if command -v systemctl &> /dev/null; then
                    sudo systemctl start redis || sudo systemctl start redis-server
                else
                    redis-server --daemonize yes
                fi
                ;;
            stop)
                echo "Stopping Redis..."
                if command -v systemctl &> /dev/null; then
                    sudo systemctl stop redis || sudo systemctl stop redis-server
                else
                    redis-cli shutdown
                fi
                ;;
            status)
                if redis-cli ping > /dev/null 2>&1; then
                    echo "Redis is running"
                    redis-cli info server | grep -E "redis_version|uptime"
                else
                    echo "Redis is not running"
                fi
                ;;
            *)
                echo "Usage: $0 redis {start|stop|status}"
                ;;
        esac
        ;;
    start)
        echo "Starting application..."

        # Check if already running
        if screen -ls | grep -q "$SCREEN_NAME"; then
            echo "Screen session already exists. Use 'restart' to restart."
            exit 1
        fi

        # Check for stale PID files
        if [ -f "$APP_DIR/fastapi_backend.pid" ] || [ -f "$APP_DIR/streamlit_frontend.pid" ] || [ -f "$APP_DIR/celery_worker.pid" ]; then
            echo "Stale PID files found. Cleaning up..."
            rm -f "$APP_DIR/fastapi_backend.pid" "$APP_DIR/streamlit_frontend.pid" "$APP_DIR/celery_worker.pid"
        fi

        # Check if run_all_linux.sh exists
        if [ ! -f "$APP_DIR/run_all_linux.sh" ]; then
            echo "Error: run_all_linux.sh not found in $APP_DIR"
            exit 1
        fi

        # Check Redis availability
        if command -v redis-cli &> /dev/null; then
            if ! redis-cli ping > /dev/null 2>&1; then
                echo "⚠ Warning: Redis is not running. Background jobs will not work."
                echo "  Start Redis with: $0 redis start"
                echo ""
            fi
        fi

        # Ensure we're in the app directory
        cd "$APP_DIR"
        
        # Start in screen session
        screen -dmS "$SCREEN_NAME" bash -c './run_all_linux.sh 2>&1 | tee ~/rpp_output.log; echo "Press enter to close"; read'

        echo "Waiting for services to start..."
        sleep 6
        
        # Verify services started
        backend_running=false
        celery_running=false
        frontend_running=false
        
        if pgrep -f "gunicorn.*api.main:app" > /dev/null; then
            backend_running=true
        fi
        
        if pgrep -f "celery.*worker" > /dev/null; then
            celery_running=true
        fi
        
        if pgrep -f "streamlit.*frontend" > /dev/null; then
            frontend_running=true
        fi
        
        if $backend_running && $frontend_running; then
            echo "✓ Application started successfully"
            echo ""
            echo "Backend:  http://localhost:8000"
            [ "$celery_running" = true ] && echo "Celery:   Running (background jobs enabled)" || echo "Celery:   Not running (background jobs disabled)"
            echo "Frontend: http://localhost:8501"
        else
            echo "⚠ Application may not have started correctly:"
            [ "$backend_running" = false ] && echo "  - Backend not detected"
            [ "$celery_running" = false ] && echo "  - Celery worker not detected"
            [ "$frontend_running" = false ] && echo "  - Frontend not detected"
            echo ""
            echo "Check logs with: $0 logs"
        fi
        
        echo ""
        echo "Commands:"
        echo "  Attach to session: $0 attach"
        echo "  View logs:         $0 logs"
        echo "  Check status:      $0 status"
        ;;
    *)
        echo "Usage: $0 {status|attach|start|restart|stop|clean|logs|redis}"
        echo ""
        echo "  status  - Show running screen sessions and processes"
        echo "  attach  - Attach to the screen session"
        echo "  start   - Start the application (fails if already running)"
        echo "  restart - Stop and restart the application"
        echo "  stop    - Stop the application gracefully"
        echo "  clean   - Force kill all related processes"
        echo "  logs [backend|celery|frontend|all] - View application logs"
        echo "  redis {start|stop|status} - Manage Redis server"
        exit 1
        ;;
esac