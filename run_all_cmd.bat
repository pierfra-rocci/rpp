@echo off
REM Activate virtual environment
call .venv\Scripts\activate

REM Setup database
set APP_ENV=development

REM -----------------------------------------------------------------------------
REM IMPORTANT: Set the SMTP password for development before running the script.
REM This should be set as an environment variable, not hardcoded here.
REM For example, in this terminal, you can run: set SMTP_PASS="your_secret_password"
REM -----------------------------------------------------------------------------
if not defined SMTP_PASS (
    echo Warning: SMTP_PASS environment variable is not set.
    echo The application may not be able to send emails.
    echo.
)

REM Start backend.py in a new terminal, redirecting output to backend.log
start "Python Backend" cmd /k "python backend.py > backend.log 2>&1"

REM Start Streamlit frontend in a new terminal
start "Streamlit Frontend" cmd /k "streamlit run frontend.py --server.port 8501 --server.address 127.0.0.1"

REM Print URLs and log file names
echo Backend URL: http://127.0.0.1:5000
echo Frontend URL: http://127.0.0.1:8501
echo Backend Log: backend.log (in its own window or file)
echo Frontend output will be in its own terminal window.
