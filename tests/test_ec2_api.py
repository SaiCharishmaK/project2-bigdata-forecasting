#!/usr/bin/env python3
"""
Test Taxi Demand Forecasting API
Tests localhost API endpoint
"""
import requests
import json
import sys

# Configuration
BASE_URL = "http://100.24.50.167:8000"

# Sample 31 features for prediction
SAMPLE_FEATURES = [
    1150, 1150, 1145, 1140, 1135,      # lag features
    1150, 1100, 1180,                  # more lag features
    1175, 1172, 1170, 1168, 25, 1175, # rolling features
    8, 2, 15, 20, 5,                   # time features
    0, 1, 0, 1, 0, 0,                  # binary time features
    0.5, 0.866, 0.707, 0.707, 0.866, 0.5  # cyclical features
]

def test_health():
    print("\n" + "="*60)
    print("🏥 Health Check")
    print("="*60)
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=10)
        data = r.json()
        print(f"Status: {r.status_code}")
        print(f"Model Loaded: {data['model_loaded']}")
        print(f"Features: {data['feature_count']}")
        assert r.status_code == 200
        print("✅ PASSED\n")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}\n")
        return False

def test_single_prediction():
    print("="*60)
    print("🎯 Single Prediction")
    print("="*60)
    try:
        payload = {"features": SAMPLE_FEATURES, "location": "zone-1"}
        r = requests.post(f"{BASE_URL}/predict", json=payload, timeout=10)
        data = r.json()
        print(f"Status: {r.status_code}")
        print(f"Predicted: {data['predicted_trip_count']:.2f} trips/hour")
        print(f"Confidence: {data['prediction_confidence']:.2%}")
        assert r.status_code == 200
        print("✅ PASSED\n")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}\n")
        return False

def test_batch_prediction():
    print("="*60)
    print("📦 Batch Prediction")
    print("="*60)
    try:
        payload = {
            "forecasts": [
                {"features": SAMPLE_FEATURES, "location": "zone-1"},
                {"features": [x*1.1 for x in SAMPLE_FEATURES], "location": "zone-2"},
                {"features": [x*0.9 for x in SAMPLE_FEATURES], "location": "zone-3"}
            ]
        }
        r = requests.post(f"{BASE_URL}/predict/batch", json=payload, timeout=10)
        data = r.json()
        print(f"Status: {r.status_code}")
        print(f"Total: {data['total_forecasts']}")
        print(f"Successful: {data['successful']}")
        print(f"Batch time: {data['batch_processing_time_ms']:.2f}ms")
        for i, p in enumerate(data['predictions'], 1):
            print(f"  Zone {i}: {p['predicted_trip_count']:.2f}")
        assert r.status_code == 200
        print("✅ PASSED\n")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}\n")
        return False

if __name__ == "__main__":
    print("\n" + "█"*60)
    print("🚀 TAXI DEMAND FORECASTING API TEST")
    print("█"*60)
    
    results = [
        test_health(),
        test_single_prediction(),
        test_batch_prediction()
    ]
    
    if all(results):
        print("="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print(f"\n📚 Interactive Docs: {BASE_URL}/docs")
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        sys.exit(1)