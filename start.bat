@echo off
echo Starting Face Recognition App...
echo.

REM Kiểm tra Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found! Please install Python 3.8+
    pause
    exit /b 1
)

REM Kiểm tra virtual environment
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Kích hoạt virtual environment
call venv\Scripts\activate.bat

REM Cài đặt dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Chạy ứng dụng
echo Starting application...
python run.py

pause
