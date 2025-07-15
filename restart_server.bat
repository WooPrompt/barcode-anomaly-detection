@echo off
echo Stopping any running uvicorn servers...
taskkill /F /IM python.exe /T 2>nul
timeout /t 2 /nobreak >nul

echo Starting FastAPI server with reload...
cd /d "%~dp0"
uvicorn fastapi_server:app --host 0.0.0.0 --port 8000 --reload

pause