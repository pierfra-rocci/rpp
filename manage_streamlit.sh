#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${SCRIPT_DIR}"
SCREEN_NAME="rpp_app"

case "$1" in
    status)
        echo "=== Screen Sessions ==="
        screen -ls | grep "$SCREEN_NAME" || echo "No screen session found"
        echo ""
        echo "=== Running Processes ==="
        echo "FastAPI/Gunicorn:"
        pgrep -fa "gunicorn.*api.main:app" || echo "  Not running"
        echo "Streamlit:"
        pgrep -fa "streamlit run" || echo "  Not running"
        echo ""
        if [ -f "$APP_DIR/fastapi_backend.pid" ]; then
            backend_pid=$(cat "$APP_DIR/fastapi_backend.pid" 2>/dev/null)
            if [ -n "$backend_pid" ] && ps -p "$backend_pid" > /dev/null 2>&1; then
                echo "Backend PID: $backend_pid (running)"
            else
                echo "Backend PID file exists but process not running"
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
        pkill -f "gunicorn.*api.main:app" 2>/dev/null || true
        pkill -f "streamlit run frontend.py" 2>/dev/null || true
        
        echo "✓ Application stopped"
        ;;
    clean)
        echo "Cleaning up ALL streamlit and backend processes..."
        pkill -9 -f "streamlit" || true
        pkill -9 -f "gunicorn" || true
        pkill -9 -f "uvicorn" || true
        screen -S "$SCREEN_NAME" -X quit 2>/dev/null || true
        rm -f "$APP_DIR/fastapi_backend.pid" "$APP_DIR/streamlit_frontend.pid"
        rm -f "$APP_DIR/fastapi_backend.log" "$APP_DIR/streamlit_frontend.log"
        echo "✓ Cleanup complete"
        ;;
    logs)
        if [ -f "$APP_DIR/fastapi_backend.log" ] || [ -f "$APP_DIR/streamlit_frontend.log" ]; then
            case "${2:-}" in
                backend|1)
                    tail -f "$APP_DIR/fastapi_backend.log"
                    ;;
                frontend|2)
                    tail -f "$APP_DIR/streamlit_frontend.log"
                    ;;
                both|3)
                    tail -f "$APP_DIR/fastapi_backend.log" "$APP_DIR/streamlit_frontend.log"
                    ;;
                *)
                    echo "Available logs:"
                    echo "  $0 logs backend   - View backend logs (follow)"
                    echo "  $0 logs frontend  - View frontend logs (follow)"
                    echo "  $0 logs both      - View both logs (follow)"
                    echo ""
                    echo "Recent backend log:"
                    tail -20 "$APP_DIR/fastapi_backend.log" 2>/dev/null || echo "  No backend log"
                    echo ""
                    echo "Recent frontend log:"
                    tail -20 "$APP_DIR/streamlit_frontend.log" 2>/dev/null || echo "  No frontend log"
                    ;;
            esac
        else
            echo "No log files found in $APP_DIR"
            if [ -f "$HOME/rpp_output.log" ]; then
                echo "Try: tail -f ~/rpp_output.log"
            fi
        fi
        ;;
    start)
        echo "Starting application..."

        # Check if already running
        if screen -ls | grep -q "$SCREEN_NAME"; then
            echo "Screen session already exists. Use 'restart' to restart."
            exit 1
        fi

        # Check for stale PID files
        if [ -f "$APP_DIR/fastapi_backend.pid" ] || [ -f "$APP_DIR/streamlit_frontend.pid" ]; then
            echo "Stale PID files found. Cleaning up..."
            rm -f "$APP_DIR/fastapi_backend.pid" "$APP_DIR/streamlit_frontend.pid"
        fi

        # Ensure we're in the app directory
        cd "$APP_DIR"
        
        # Start in screen session with proper shell handling
        screen -dmS "$SCREEN_NAME" bash -c "cd '$APP_DIR' && ./run_all_linux.sh 2>&1 | tee ~/rpp_output.log; echo 'Services stopped. Press enter to close.'; read"

        echo "Waiting for services to start..."
        sleep 8
        
        # Verify services started
        backend_running=false
        frontend_running=false
        
        if pgrep -f "gunicorn.*api.main:app" > /dev/null; then
            backend_running=true
        fi
        
        if pgrep -f "streamlit run frontend.py" > /dev/null; then
            frontend_running=true
        fi
        
        if $backend_running && $frontend_running; then
            echo "✓ Application started successfully"
            echo ""
            echo "Backend:  http://localhost:8000"
            echo "Frontend: http://localhost:8501"
        else
            echo "⚠ Application may not have started correctly:"
            [ "$backend_running" = false ] && echo "  - Backend not detected"
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
        echo "Usage: $0 {status|attach|start|restart|stop|clean|logs}"
        echo ""
        echo "  status  - Show running screen sessions and processes"
        echo "  attach  - Attach to the screen session"
        echo "  start   - Start the application (fails if already running)"
        echo "  restart - Stop and restart the application"
        echo "  stop    - Stop the application gracefully"
        echo "  clean   - Force kill all related processes"
        echo "  logs [backend|frontend|both] - View application logs"
        exit 1
        ;;
esac