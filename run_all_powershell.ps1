# Activate virtual environment
. .\.venv\Scripts\Activate.ps1

# Start backend_dev.py as a background job, redirecting output to backend.log
Start-Job -ScriptBlock { python backend_dev.py *>&1 | Tee-Object -FilePath backend.log }

# Start run_frontend.py as a background job, redirecting output to frontend.log
Start-Job -ScriptBlock { python run_frontend.py *>&1 | Tee-Object -FilePath frontend.log }

Write-Host "Both backend and frontend are running as background jobs."
Write-Host "Backend URL: https://127.0.0.1:5000"
Write-Host "Frontend URL: https://127.0.0.1:8501"
Write-Host "Logs: backend.log, frontend.log"

# Get-Job | Remove-Job