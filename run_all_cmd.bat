@echo off
setlocal

REM Windows helper for local development and testing only.

if not exist .venv\Scripts\activate (
	echo Virtual environment not found. Please create .venv first.
	exit /b 1
)

call .venv\Scripts\activate

REM Development environment variables
set APP_ENV=development
set RPP_API_URL=http://127.0.0.1:8000
set RPP_LEGACY_URL=http://127.0.0.1:5000

REM Start FastAPI backend with auto-reload
start "FastAPI Backend" cmd /k "python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000"

REM Launch Streamlit frontend
start "Streamlit Frontend" cmd /k "streamlit run frontend.py --server.port 8501 --server.address 127.0.0.1"

echo.
echo FastAPI backend: http://127.0.0.1:8000
echo Streamlit app:   http://127.0.0.1:8501
echo Close the spawned terminals or press CTRL+C in each to stop services.
echo.

endlocal
