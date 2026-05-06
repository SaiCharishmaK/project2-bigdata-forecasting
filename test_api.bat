@echo off
REM Test API with curl

REM Change this URL to test EC2
set API_URL=http://localhost:8000
REM set API_URL=http://100.24.50.167:8000

echo Testing Health Endpoint...
curl -X GET "%API_URL%/health" -H "Content-Type: application/json"
echo.
echo.

echo Testing Single Prediction...
curl -X POST "%API_URL%/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"features\":[10.5,20.3,15.1,12.0,18.5,1150.0,1100.0,1180.0,1175.0,1172.0,1170.0,1168.0,25.0,1175.0,8.0,2.0,15.0,20.0,5.0,0.0,1.0,0.0,1.0,0.0,0.0,0.5,0.866,0.707,0.707,0.866,0.5],\"location\":\"zone-1\"}"
echo.
echo.

echo Opening API Documentation in browser...
start "%API_URL%/docs"
