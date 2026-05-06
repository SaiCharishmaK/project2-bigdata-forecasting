#!/usr/bin/env python3
"""
Test script for Taxi Demand Forecasting API
"""
import requests
import json

# API endpoint
API_URL = "http://localhost:8000"  # Change to http://100.24.50.167:8000 for EC2

# Sample 31 features for prediction
# Features: lag_1, lag_2, lag_3, lag_6, lag_12, lag_24, lag_48, lag_168,
#          rolling_mean_3h, rolling_mean_6h, rolling_mean_12h, rolling_mean_24h, rolling_std_24h, rolling_mean_168h,
#          hour_of_day, day_of_week, day_of_month, week_of_year, month,
#          is_weekend, is_rush_hour, is_night, is_morning, is_afternoon, is_evening,
#          hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos

sample_features = [
    10.5,    # lag_1
    20.3,    # lag_2
    15.1,    # lag_3
    12.0,    # lag_6
    18.5,    # lag_12
    1150.0,  # lag_24 (from your example)
    1100.0,  # lag_48
    1180.0,  # lag_168 (from your example)
    1175.0,  # rolling_mean_3h
    1172.0,  # rolling_mean_6h
    1170.0,  # rolling_mean_12h
    1168.0,  # rolling_mean_24h
    25.0,    # rolling_std_24h
    1175.0,  # rolling_mean_168h
    8.0,     # hour_of_day (from your example)
    2.0,     # day_of_week (from your example)
    15.0,    # day_of_month
    20.0,    # week_of_year
    5.0,     # month
    0.0,     # is_weekend (from your example)
    1.0,     # is_rush_hour
    0.0,     # is_night
    1.0,     # is_morning
    0.0,     # is_afternoon
    0.0,     # is_evening
    0.5,     # hour_sin
    0.866,   # hour_cos
    0.707,   # dow_sin
    0.707,   # dow_cos
    0.866,   # month_sin
    0.5      # month_cos
]

def test_health():
    """Test health endpoint"""
    print("=" * 60)
    print("🏥 Testing Health Endpoint")
    print("=" * 60)
    try:
        response = requests.get(f"{API_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        print()
    except Exception as e:
        print(f"❌ Error: {e}\n")

def test_single_prediction():
    """Test single prediction"""
    print("=" * 60)
    print("🎯 Testing Single Prediction")
    print("=" * 60)
    
    payload = {
        "features": sample_features,
        "location": "zone-1"
    }
    
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        print()
    except Exception as e:
        print(f"❌ Error: {e}\n")

def test_batch_prediction():
    """Test batch prediction"""
    print("=" * 60)
    print("📦 Testing Batch Prediction (3 forecasts)")
    print("=" * 60)
    
    payload = {
        "forecasts": [
            {
                "features": sample_features,
                "location": "zone-1"
            },
            {
                "features": [x * 1.1 for x in sample_features],  # Slightly different values
                "location": "zone-2"
            },
            {
                "features": [x * 0.9 for x in sample_features],  # Slightly different values
                "location": "zone-3"
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{API_URL}/predict/batch",
            json=payload
        )
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Total: {result['total_forecasts']}")
        print(f"Successful: {result['successful']}")
        print(f"Failed: {result['failed']}")
        print(f"Batch time: {result['batch_processing_time_ms']:.2f}ms")
        print(f"\nFirst prediction: {result['predictions'][0]['predicted_trip_count']:.2f} trips/hour")
        print()
    except Exception as e:
        print(f"❌ Error: {e}\n")

def test_documentation():
    """Show API documentation URL"""
    print("=" * 60)
    print("📚 API Documentation")
    print("=" * 60)
    print(f"Interactive Docs: {API_URL}/docs")
    print(f"OpenAPI Schema: {API_URL}/openapi.json")
    print()

if __name__ == "__main__":
    print("\n🚀 Taxi Demand Forecasting API - Test Suite\n")
    
    test_health()
    test_single_prediction()
    test_batch_prediction()
    test_documentation()
    
    print("✅ All tests completed!")
