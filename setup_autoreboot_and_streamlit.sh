echo "✓ Application stopped"
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
echo "✓ Weekly reboot scheduled: Every Monday at midnight"
echo "✓ Auto-start service enabled: Streamlit will start after reboot"
echo "✓ Management script created: $MGMT_SCRIPT"
echo ""
echo "=== Quick Commands ==="
echo "View screen sessions:   ~/manage_streamlit.sh status"
echo "Attach to session:      ~/manage_streamlit.sh attach"
echo "Restart application:    ~/manage_streamlit.sh restart"
echo "Stop application:       ~/manage_streamlit.sh stop"
echo "Clean all processes:    ~/manage_streamlit.sh clean"
echo "View logs:              ~/manage_streamlit.sh logs"
echo ""
echo "=== Cron Schedule ==="
crontab -l | grep reboot || echo "(No reboot cron found)"
echo ""
echo "=== Test the service now (optional) ==="
echo "sudo systemctl start streamlit-autostart.service"
echo "~/manage_streamlit.sh status"
echo ""