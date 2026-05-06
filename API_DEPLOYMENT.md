# API Deployment Guide

## Local Development

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the API
**Windows:**
```bash
run_api.bat
```

**Linux/Mac:**
```bash
bash run_api.sh
```

### 3. Test the API
```bash
# Health check
curl http://localhost:8000/health

# Single prediction (31 features required)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [10.5, 20.3, 15.1, 12.0, 18.5, 25.0, 30.0, 35.2, 8.1, 12.3, 15.4, 18.5, 5.0, 40.1, 0, 1, 15, 20, 5, 0, 0, 0, 1, 0, 0, 0.5, 0.866, 0.707, 0.707, 0.866, 0.5],
    "location": "zone-1"
  }'

# View API docs
# Open: http://localhost:8000/docs
```

## EC2 Deployment

### 1. Connect to EC2
```bash
ssh -i ~/.ssh/project2-key.pem ec2-user@100.24.50.167
```

### 2. Clone/Upload Repository
```bash
git clone <repo-url> ml-portfolio
cd ml-portfolio/project2-bigdata-forecasting
```

### 3. Setup Python Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Verify Model Files
```bash
ls models/
# Should show: xgb_champion.json or models/sagemaker/xgb_model.json
```

### 5. Start API with Systemd (Recommended)

**Create `/etc/systemd/system/taxi-forecast-api.service`:**
```ini
[Unit]
Description=Taxi Demand Forecasting API
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/ml-portfolio/project2-bigdata-forecasting
Environment="PATH=/home/ec2-user/ml-portfolio/project2-bigdata-forecasting/venv/bin"
ExecStart=/home/ec2-user/ml-portfolio/project2-bigdata-forecasting/venv/bin/python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Enable and start:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable taxi-forecast-api
sudo systemctl start taxi-forecast-api
```

### 6. Monitor Logs
```bash
# View logs
sudo journalctl -u taxi-forecast-api -f

# Check status
sudo systemctl status taxi-forecast-api
```

### 7. Allow Port 8000 in Security Group
- Go to AWS Security Groups
- Add inbound rule: Port 8000, Source: 0.0.0.0/0 (or restrict to your IPs)

### 8. Test Deployment
```bash
curl http://100.24.50.167:8000/health
```

## Endpoints

### Health Check
- **GET** `/health` - Basic health status
- **GET** `/health/detailed` - Detailed health info

### Predictions
- **POST** `/predict` - Single forecast (31 features)
- **POST** `/predict/batch` - Batch forecast (up to 100 requests)

### Documentation
- **GET** `/docs` - Interactive API documentation (Swagger UI)
- **GET** `/openapi.json` - OpenAPI schema

## Feature Engineering

The model expects 31 features:
1. **Lag features** (8): lag_1, lag_2, lag_3, lag_6, lag_12, lag_24, lag_48, lag_168
2. **Rolling features** (6): rolling_mean_3h, rolling_mean_6h, rolling_mean_12h, rolling_mean_24h, rolling_std_24h, rolling_mean_168h
3. **Time features** (11): hour_of_day, day_of_week, day_of_month, week_of_year, month, is_weekend, is_rush_hour, is_night, is_morning, is_afternoon, is_evening
4. **Cyclical features** (6): hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos

Total: 31 features

## Performance Expectations

- **Inference time**: ~1-2ms per prediction
- **Batch processing**: ~50-100 predictions/second
- **Memory usage**: ~500MB-1GB
- **Latency**: P99 < 100ms

## Monitoring & Logging

Health check endpoint returns:
- Model load status
- Feature count
- Device (CPU/GPU)
- Model version
- Timestamp

## Troubleshooting

**Port 8000 already in use:**
```bash
lsof -i :8000  # Find process
kill -9 <PID>  # Kill process
```

**Model not found:**
```bash
# Check if model exists
ls -la models/
# Download from S3 if needed
aws s3 cp s3://document-ai-portfolio-data/project2/sagemaker/xgb_model.json models/
```

**Permission denied on startup:**
```bash
chmod +x run_api.sh
chmod +x run_api.bat
```
