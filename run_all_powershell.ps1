# Activate virtual environment
. .\venv\Scripts\Activate.ps1

# Start backend_dev.py as a background job
Start-Job -ScriptBlock { python backend_dev.py }

# Start run_frontend.py as a background job
Start-Job -ScriptBlock { python run_frontend.py }

Write-Host "Both backend and frontend are running as background jobs."
