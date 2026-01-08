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
            echo "Backend PID file: $(cat $APP_DIR/fastapi_backend.pid)"
        fi
        if [ -f "$APP_DIR/streamlit_frontend.pid" ]; then
            echo "Frontend PID file: $(cat $APP_DIR/streamlit_frontend.pid)"
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
        sleep 2
        $0 start
        ;;
    stop)
        echo "Stopping application..."
        
        # Stop via screen session (run_all_linux.sh cleanup trap will handle graceful shutdown)
        if screen -ls | grep -q "$SCREEN_NAME"; then
            screen -S "$SCREEN_NAME" -X quit 2>/dev/null || true
            sleep 3
        fi
        
        # Fallback: kill processes using PID files
        if [ -f "$APP_DIR/fastapi_backend.pid" ]; then
            backend_pid=$(cat "$APP_DIR/fastapi_backend.pid")
            if ps -p "$backend_pid" > /dev/null 2>&1; then
                kill "$backend_pid" 2>/dev/null || true
                sleep 1
                kill -9 "$backend_pid" 2>/dev/null || true
            fi
            rm -f "$APP_DIR/fastapi_backend.pid"
        fi
        
        if [ -f "$APP_DIR/streamlit_frontend.pid" ]; then
            frontend_pid=$(cat "$APP_DIR/streamlit_frontend.pid")
            if ps -p "$frontend_pid" > /dev/null 2>&1; then
                kill "$frontend_pid" 2>/dev/null || true
                sleep 1
                kill -9 "$frontend_pid" 2>/dev/null || true
            fi
            rm -f "$APP_DIR/streamlit_frontend.pid"
        fi
        
        # Final cleanup
        pkill -f "gunicorn.*api.main:app" 2>/dev/null || true
        pkill -f "streamlit run" 2>/dev/null || true
        
        echo "✓ Application stopped"
        ;;
    clean)
        echo "Cleaning up ALL streamlit and backend processes..."
        pkill -9 -f "streamlit" || true
        pkill -9 -f "gunicorn" || true
        pkill -9 -f "uvicorn" || true
        screen -S "$SCREEN_NAME" -X quit 2>/dev/null || true
        rm -f "$APP_DIR/fastapi_backend.pid" "$APP_DIR/streamlit_frontend.pid"
        echo "✓ Cleanup complete"
        ;;
    logs)
        if [ -f "$APP_DIR/fastapi_backend.log" ] || [ -f "$APP_DIR/streamlit_frontend.log" ]; then
            echo "Choose log to view:"
            echo "  1) Backend (FastAPI/Gunicorn)"
            echo "  2) Frontend (Streamlit)"
            echo "  3) Both (split view)"
            read -p "Selection [1-3]: " choice
            case $choice in
                1)
                    tail -f "$APP_DIR/fastapi_backend.log"
                    ;;
                2)
                    tail -f "$APP_DIR/streamlit_frontend.log"
                    ;;
                3)
                    tail -f "$APP_DIR/fastapi_backend.log" "$APP_DIR/streamlit_frontend.log"
                    ;;
                *)
                    echo "Invalid choice"
                    exit 1
                    ;;
            esac
        else
            echo "No log files found. Try: tail -f ~/rpp_output.log"
        fi
        ;;
    start)
        echo "Starting application..."

        # Check if already running
        if screen -ls | grep -q "$SCREEN_NAME"; then
            echo "Screen session already exists. Use 'restart' to restart."
            exit 1
        fi

        # Ensure we're in the app directory
        cd "$APP_DIR"
        
        # Start in screen session
        screen -dmS "$SCREEN_NAME" bash -c './run_all_linux.sh 2>&1 | tee ~/rpp_output.log; echo "Press enter to close"; read'

        sleep 5
        
        # Verify services started
        if pgrep -f "gunicorn.*api.main:app" > /dev/null && pgrep -f "streamlit run" > /dev/null; then
            echo "✓ Application started successfully"
        else
            echo "⚠ Application may not have started correctly. Check logs."
        fi
        
        echo "Attach to session: $0 attach"
        echo "View logs: $0 logs"
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
        echo "  logs    - View application logs (backend/frontend)"
        exit 1
        ;;
esac