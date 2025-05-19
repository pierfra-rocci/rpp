# Activate virtual environment
. .\.venv\Scripts\Activate.ps1

# Start backend_dev.py as a background job, redirecting output to backend.log
Start-Job -ScriptBlock { python backend.py *>&1 | Tee-Object -FilePath backend.log }

# Start run_frontend.py as a background job, redirecting output to frontend.log
# Original line
# Start-Job -FilePath .\run_frontend.py -ArgumentList $args *>&1 | Tee-Object -FilePath frontend.log
Start-Job -ScriptBlock { streamlit run frontend.py --server.port 8501 --server.address 127.0.0.1 } *>&1 | Tee-Object -FilePath frontend.log

Write-Host "Both backend and frontend are running as background jobs."
Write-Host "Backend URL: http://127.0.0.1:5000"
Write-Host "Frontend URL: http://127.0.0.1:80"
Write-Host "Logs: backend.log, frontend.log"

# Get-Job | Remove-Job