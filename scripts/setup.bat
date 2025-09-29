@echo off

REM Lead HeatScore - Windows Setup Script
REM Sets up the complete RAG-powered Lead HeatScore system

echo.
echo  Lead HeatScore Setup Starting...

REM Check for Python 3.10+
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

REM Check for Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found. Please install Node.js 18+
    pause
    exit /b 1
)

echo [INFO] Prerequisites check passed

REM Create backend environment
if not exist "backend\.env" (
    copy "backend\env.example" "backend\.env"
    echo.
    echo [WARNING] Environment file created at backend\.env
    echo [WARNING] Please update these required variables:
    echo   OPENAI_API_KEY=sk-your-openai-key-here
    echo   MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/
    echo   MONGO_DB=leadheat
    echo.
    pause
)

REM Setup backend
echo [INFO] Setting up backend...
cd backend

REM Create virtual environment
if not exist "venv" (
    echo [INFO] Creating Python virtual environment...
    python -m venv venv
)

REM Activate virtual environment and install dependencies
call venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

echo [SUCCESS] Backend dependencies installed
cd ..

REM Setup frontend
echo [INFO] Setting up frontend...
cd frontend
npm install
echo [SUCCESS] Frontend dependencies installed
cd ..

REM Build frontend
echo [INFO] Building frontend...
cd frontend
npm run build
echo [SUCCESS] Frontend built successfully
cd ..

echo.
echo [SUCCESS]  Setup completed successfully!
echo.
echo Next steps:
echo 1. Update backend\.env with your API keys
echo 2. Start backend: cd backend ^&^& venv\Scripts\activate ^&^& uvicorn app.main:app --reload
echo 3. Start frontend: cd frontend ^&^& npm run dev
echo 4. Open http://localhost:5173
echo.
echo  Documentation: docs\API_DOCUMENTATION.md
echo Interactive Testing: http://localhost:8000/docs (Swagger UI)
echo.

pause
