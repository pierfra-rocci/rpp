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

if ! command -v tmux &> /dev/null; then
    echo "tmux not found. Please install tmux or use the background jobs version."
    exit 1
fi

echo "Starting backend and frontend in a tmux session..."

tmux new-session -d -s rpp_session "source .venv/bin/activate; python backend_dev.py | tee $backend_log"
tmux split-window -h -t rpp_session "source .venv/bin/activate; streamlit run frontend.py | tee $frontend_log"
tmux select-layout -t rpp_session even-horizontal
tmux attach -t rpp_session
