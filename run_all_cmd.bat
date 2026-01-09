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

REM Redis/Celery configuration
set REDIS_URL=redis://localhost:6379/0
set CELERY_BROKER_URL=%REDIS_URL%
set CELERY_RESULT_BACKEND=%REDIS_URL%

REM Check if Redis is running (optional)
echo Checking Redis availability...
redis-cli ping >nul 2>&1
if %errorlevel% equ 0 (
    echo Redis is available - background jobs enabled
    set REDIS_AVAILABLE=1
) else (
    echo Redis is not running - background jobs disabled
    echo To enable background jobs, install and start Redis:
    echo   - Download from https://github.com/microsoftarchive/redis/releases
    echo   - Or use Docker: docker run -d -p 6379:6379 redis
    set REDIS_AVAILABLE=0
)

REM Start FastAPI backend with auto-reload
start "FastAPI Backend" cmd /k "python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000"

REM Start Celery worker if Redis is available
if "%REDIS_AVAILABLE%"=="1" (
    echo Starting Celery worker...
    start "Celery Worker" cmd /k "celery -A celery_app worker --loglevel=info --pool=solo"
)

REM Launch Streamlit frontend
start "Streamlit Frontend" cmd /k "streamlit run frontend.py --server.port 8501 --server.address 127.0.0.1"

echo.
echo ========================================
echo Services Started:
echo ========================================
echo FastAPI backend: http://127.0.0.1:8000
if "%REDIS_AVAILABLE%"=="1" (
    echo Celery worker:   Running (background jobs enabled)
) else (
    echo Celery worker:   Not running (Redis unavailable)
)
echo Streamlit app:   http://127.0.0.1:8501
echo.
echo Close the spawned terminals or press CTRL+C in each to stop services.
echo.

endlocal
