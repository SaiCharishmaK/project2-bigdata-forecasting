# api/app.py
"""
FastAPI application for taxi demand forecasting.
Serves XGBoost model predictions via REST API.
"""
import time
import json
from datetime import datetime
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.models import (
    ForecastRequest,
    BatchForecastRequest,
    ForecastResponse,
    BatchForecastResponse
)
from api.predictor import predictor
from api.health import router as health_router


# ── Lifespan — runs at startup/shutdown ────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load model at startup.
    This runs ONCE when API starts.
    """
    print("\n🚀 Starting Taxi Forecast API...")
    predictor.load_models()
    print("✅ API ready to serve requests\n")
    yield
    print("Shutting down API...")


# ── Create FastAPI App ─────────────────────────
app = FastAPI(
    title="Taxi Demand Forecasting API",
    description=(
        "XGBoost-powered 24-hour taxi demand "
        "forecast API.\n\n"
        "Uses lag features, rolling statistics, "
        "and temporal features to predict "
        "taxi trip counts."
    ),
    version="1.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# ── CORS Middleware ────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Include routers ────────────────────────────
app.include_router(health_router, tags=["Health"])


# ── Main prediction endpoint ───────────────────
@app.post(
    "/predict",
    response_model=ForecastResponse,
    summary="Single Forecast",
    tags=["Predictions"]
)
async def predict_single(request: ForecastRequest):
    """
    Make a single demand forecast.
    
    **Input:**
    - features: List of 31 features
      (lag, rolling, time, cyclical)
    - location: Optional zone identifier
    
    **Output:**
    - predicted_trip_count: 24-hour forecast
    - prediction_confidence: 0-1 confidence
    - timestamp: UTC timestamp
    """
    try:
        start_time = time.time()
        
        # Validate features
        if not predictor.validate_features(
                request.features):
            raise ValueError(
                f"Expected 31 features, "
                f"got {len(request.features)}")
        
        # Make prediction
        import numpy as np
        features_array = np.array(
            request.features, dtype=np.float32)
        
        prediction, confidence = predictor.predict(
            features_array)
        
        processing_time = \
            (time.time() - start_time) * 1000
        
        return ForecastResponse(
            predicted_trip_count=prediction,
            prediction_confidence=confidence,
            location=request.location,
            model_version=predictor.model_version,
            processing_time_ms=processing_time,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}")


@app.post(
    "/predict/batch",
    response_model=BatchForecastResponse,
    summary="Batch Forecast",
    tags=["Predictions"]
)
async def predict_batch(
        request: BatchForecastRequest):
    """
    Make multiple demand forecasts.
    
    **Input:**
    - forecasts: List of up to 100 
      forecast requests
    
    **Output:**
    - predictions: List of forecasts
    - total_forecasts: Number requested
    - successful: Number completed
    - failed: Number that failed
    """
    try:
        import numpy as np
        start_time = time.time()
        
        predictions = []
        failed_count = 0
        
        for i, forecast_req \
                in enumerate(request.forecasts):
            try:
                # Validate
                if not predictor.validate_features(
                        forecast_req.features):
                    failed_count += 1
                    continue
                
                # Predict
                features_array = np.array(
                    forecast_req.features,
                    dtype=np.float32)
                
                prediction, confidence = \
                    predictor.predict(
                        features_array)
                
                predictions.append(
                    ForecastResponse(
                        predicted_trip_count=
                            prediction,
                        prediction_confidence=
                            confidence,
                        location=forecast_req.location,
                        model_version=
                            predictor.model_version,
                        processing_time_ms=0.0,
                        timestamp=
                            datetime.utcnow().isoformat()
                    )
                )
                
            except Exception:
                failed_count += 1
        
        batch_time = \
            (time.time() - start_time) * 1000
        
        return BatchForecastResponse(
            total_forecasts=len(
                request.forecasts),
            successful=len(predictions),
            failed=failed_count,
            predictions=predictions,
            batch_processing_time_ms=batch_time,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Batch prediction failed: "
                   f"{str(e)}")


# ── Root endpoint ──────────────────────────────
@app.get(
    "/",
    tags=["Info"]
)
async def root():
    """API information and documentation link."""
    return {
        "title": "Taxi Demand Forecasting API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": predictor.is_loaded,
        "documentation": "/docs",
        "health": "/health"
    }


# ── Error handlers ─────────────────────────────
@app.exception_handler(ValueError)
async def value_error_handler(request: Request,
                              exc: ValueError):
    """Handle validation errors gracefully."""
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )
