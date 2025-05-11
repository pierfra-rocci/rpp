@echo off
REM Activate virtual environment
call .venv\Scripts\activate

REM Start backend_dev.py in a new terminal, redirecting output to backend.log
start "Python Backend" cmd /k "python backend_dev.py > backend.log 2>&1"

REM Start Streamlit frontend in a new terminal
start "Streamlit Frontend" cmd /k "streamlit run frontend.py --server.port 8501 --server.address 127.0.0.1"

REM Print URLs and log file names
echo Backend URL: http://127.0.0.1:5000
echo Frontend URL: http://127.0.0.1:8501
echo Backend Log: backend.log (in its own window or file)
echo Frontend output will be in its own terminal window.
