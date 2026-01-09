#!/bin/bash

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (use sudo)"
    exit 1
fi

# Get the actual user who invoked sudo
ACTUAL_USER="${SUDO_USER:-$USER}"
if [ "$ACTUAL_USER" = "root" ]; then
    echo "Error: Cannot determine non-root user. Please run with: sudo -u username $0"
    exit 1
fi

APP_DIR="/home/$ACTUAL_USER/rpp"

# 1. Install Redis if not present
echo "=== Checking Redis Installation ==="
if ! command -v redis-server &> /dev/null; then
    echo "Installing Redis..."
    if command -v apt-get &> /dev/null; then
        apt-get update && apt-get install -y redis-server
    elif command -v yum &> /dev/null; then
        yum install -y redis
    elif command -v dnf &> /dev/null; then
        dnf install -y redis
    else
        echo "Warning: Could not install Redis automatically. Please install manually."
    fi
fi

# Enable and start Redis
if command -v systemctl &> /dev/null; then
    echo "Enabling Redis service..."
    systemctl enable redis-server 2>/dev/null || systemctl enable redis 2>/dev/null || true
    systemctl start redis-server 2>/dev/null || systemctl start redis 2>/dev/null || true
fi

# Verify Redis is running
if redis-cli ping > /dev/null 2>&1; then
    echo "✓ Redis is running"
else
    echo "⚠ Redis is not responding. Background jobs may not work."
fi

# ...existing code for sections 1-4...

# 5. Create management script
MGMT_SCRIPT="$APP_DIR/manage_streamlit.sh"
cat > "$MGMT_SCRIPT" << 'MGMT_EOF'
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
        echo "Celery Worker:"
        pgrep -fa "celery.*worker" || echo "  Not running"
        echo "Streamlit:"
        pgrep -fa "streamlit run" || echo "  Not running"
        echo ""
        echo "=== Redis Status ==="
        if redis-cli ping > /dev/null 2>&1; then
            echo "Redis: ✓ Running"
        else
            echo "Redis: ✗ Not responding"
        fi
        echo ""
        if [ -f "$APP_DIR/fastapi_backend.pid" ]; then
            echo "Backend PID file: $(cat $APP_DIR/fastapi_backend.pid)"
        fi
        if [ -f "$APP_DIR/celery_worker.pid" ]; then
            echo "Celery PID file: $(cat $APP_DIR/celery_worker.pid)"
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
        
        if screen -ls | grep -q "$SCREEN_NAME"; then
            screen -S "$SCREEN_NAME" -X quit 2>/dev/null || true
            sleep 3
        fi
        
        # Stop Celery worker
        if [ -f "$APP_DIR/celery_worker.pid" ]; then
            celery_pid=$(cat "$APP_DIR/celery_worker.pid")
            if ps -p "$celery_pid" > /dev/null 2>&1; then
                kill "$celery_pid" 2>/dev/null || true
                sleep 1
                kill -9 "$celery_pid" 2>/dev/null || true
            fi
            rm -f "$APP_DIR/celery_worker.pid"
        fi
        
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
        
        pkill -f "celery.*worker.*celery_app" 2>/dev/null || true
        pkill -f "gunicorn.*api.main:app" 2>/dev/null || true
        pkill -f "streamlit run" 2>/dev/null || true
        
        echo "✓ Application stopped"
        ;;
    clean)
        echo "Cleaning up ALL processes..."
        pkill -9 -f "streamlit" || true
        pkill -9 -f "gunicorn" || true
        pkill -9 -f "uvicorn" || true
        pkill -9 -f "celery" || true
        screen -S "$SCREEN_NAME" -X quit 2>/dev/null || true
        rm -f "$APP_DIR/fastapi_backend.pid" "$APP_DIR/streamlit_frontend.pid" "$APP_DIR/celery_worker.pid"
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
                    echo "  $0 logs backend   - View backend logs"
                    echo "  $0 logs celery    - View Celery worker logs"
                    echo "  $0 logs frontend  - View frontend logs"
                    echo "  $0 logs all       - View all logs"
                    echo ""
                    echo "Recent backend log:"
                    tail -20 "$APP_DIR/fastapi_backend.log" 2>/dev/null || echo "  No backend log"
                    echo ""
                    echo "Recent Celery log:"
                    tail -20 "$APP_DIR/celery_worker.log" 2>/dev/null || echo "  No Celery log"
                    echo ""
                    echo "Recent frontend log:"
                    tail -20 "$APP_DIR/streamlit_frontend.log" 2>/dev/null || echo "  No frontend log"
                    ;;
            esac
        else
            echo "No log files found. Try: tail -f ~/rpp_output.log"
        fi
        ;;
    redis)
        case "${2:-}" in
            start)
                sudo systemctl start redis-server || sudo systemctl start redis
                ;;
            stop)
                sudo systemctl stop redis-server || sudo systemctl stop redis
                ;;
            status)
                redis-cli ping && redis-cli info server | grep -E "redis_version|uptime"
                ;;
            *)
                echo "Usage: $0 redis {start|stop|status}"
                ;;
        esac
        ;;
    start)
        echo "Starting application..."

        if screen -ls | grep -q "$SCREEN_NAME"; then
            echo "Screen session already exists. Use 'restart' to restart."
            exit 1
        fi

        # Check if run_all_linux.sh exists
        if [ ! -f "$APP_DIR/run_all_linux.sh" ]; then
            echo "Error: run_all_linux.sh not found in $APP_DIR"
            exit 1
        fi

        # Check Redis
        if ! redis-cli ping > /dev/null 2>&1; then
            echo "⚠ Warning: Redis not running. Background jobs disabled."
        fi

        cd "$APP_DIR"
        screen -dmS "$SCREEN_NAME" bash -c './run_all_linux.sh 2>&1 | tee ~/rpp_output.log; echo "Press enter to close"; read'

        sleep 6
        
        if pgrep -f "gunicorn.*api.main:app" > /dev/null && pgrep -f "streamlit run" > /dev/null; then
            echo "✓ Application started successfully"
            pgrep -f "celery.*worker" > /dev/null && echo "✓ Celery worker running" || echo "⚠ Celery worker not running"
        else
            echo "⚠ Application may not have started correctly. Check logs."
        fi
        
        echo "Attach to session: $0 attach"
        echo "View logs: $0 logs"
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
MGMT_EOF

chmod +x "$MGMT_SCRIPT"
chown "$ACTUAL_USER:$ACTUAL_USER" "$MGMT_SCRIPT"
echo "✓ Created management script: $MGMT_SCRIPT"

# 6. Summary
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                   SETUP COMPLETE!                          ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "✓ Redis installed and enabled"
echo "✓ Weekly reboot scheduled: Every Monday at midnight"
echo "✓ Auto-start service enabled: Streamlit will start after reboot"
echo "✓ Management script created: $MGMT_SCRIPT"
echo ""
echo "=== Quick Commands ==="
echo "View screen sessions:   ~/rpp/manage_streamlit.sh status"
echo "Attach to session:      ~/rpp/manage_streamlit.sh attach"
echo "Restart application:    ~/rpp/manage_streamlit.sh restart"
echo "Stop application:       ~/rpp/manage_streamlit.sh stop"
echo "Clean all processes:    ~/rpp/manage_streamlit.sh clean"
echo "View logs:              ~/rpp/manage_streamlit.sh logs"
echo "Manage Redis:           ~/rpp/manage_streamlit.sh redis {start|stop|status}"
echo ""
echo "=== Background Jobs ==="
echo "Redis status: $(redis-cli ping 2>/dev/null && echo 'Running' || echo 'Not running')"
echo "Celery workers will start automatically with the application."
echo ""
echo "=== Cron Schedule ==="
crontab -l | grep reboot || echo "(No reboot cron found)"
echo ""
echo "=== Test the service now (optional) ==="
echo "sudo systemctl start streamlit-autostart.service"
echo "~/rpp/manage_streamlit.sh status"
echo ""