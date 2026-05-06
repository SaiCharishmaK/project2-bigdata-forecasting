@echo off
REM run_api.bat - Start the Taxi Demand Forecasting API (Windows)

echo.
echo Taxi Demand Forecasting API Startup
echo =====================================
echo.

REM Check if model exists
if not exist "models\xgb_champion.json" (
    if not exist "models\sagemaker\xgb_model.json" (
        echo Error: Model not found!
        echo Expected: models\xgb_champion.json
        echo Or: models\sagemaker\xgb_model.json
        exit /b 1
    )
)

echo Model found
echo.
echo Starting API server on http://0.0.0.0:8000
echo Documentation: http://localhost:8000/docs
echo Health check: http://localhost:8000/health
echo.

REM Activate venv if available
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Run API
python -m uvicorn api.app:app ^
    --host 0.0.0.0 ^
    --port 8000 ^
    --reload
