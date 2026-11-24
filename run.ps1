# Starts the MindSense backend (port 8000) and frontend (port 5173) in two windows.
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\backend'; `$env:HF_HUB_DISABLE_SYMLINKS_WARNING='1'; .venv\Scripts\python -m uvicorn app.main:app --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\frontend'; npm run dev"
Write-Host "Backend: http://localhost:8000/docs   Frontend: http://localhost:3004"
