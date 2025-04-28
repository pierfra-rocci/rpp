@echo off
REM Activate virtual environment
call .venv\Scripts\activate

REM Start backend_dev.py in a new terminal
start cmd /k "python backend_dev.py"

REM Start run_frontend.py in a new terminal
start cmd /k "python run_frontend.py"
