@echo off
REM Activate virtual environment
call .venv\Scripts\activate

REM Start backend_dev.py in a new terminal, redirecting output to backend.log
start cmd /k "python backend_dev.py > backend.log 2>&1"

REM Start run_frontend.py in a new terminal, redirecting output to frontend.log
start cmd /k "python run_frontend.py > frontend.log 2>&1"

REM Print URLs and log file names
echo Backend URL: http://127.0.0.1:5000
echo Frontend URL: http://127.0.0.1:8501
echo Logs: backend.log, frontend.log
