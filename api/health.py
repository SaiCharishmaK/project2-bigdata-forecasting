# api/health.py
"""
Health check endpoints for load balancers and monitoring.
"""
from fastapi import APIRouter
from datetime import datetime
from api.models import HealthResponse
from api.predictor import predictor

router = APIRouter()

@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"]
)
async def health_check():
    """
    Health check endpoint.
    Used by load balancers and monitoring systems.
    """
    return HealthResponse(
        status="healthy" if predictor.is_loaded 
               else "loading",
        model_loaded=predictor.is_loaded,
        model_name="XGBoost Taxi Demand Forecasting",
        model_version=predictor.model_version,
        device=predictor.device,
        timestamp=datetime.utcnow().isoformat(),
        feature_count=predictor.feature_count
    )

@router.get(
    "/health/detailed",
    tags=["Health"]
)
async def health_detailed():
    """
    Detailed health information including
    model load time and feature count.
    """
    return {
        "status": "healthy" if predictor.is_loaded 
                  else "loading",
        "model_loaded": predictor.is_loaded,
        "model_name": "XGBoost Taxi Demand Forecasting",
        "model_version": predictor.model_version,
        "device": predictor.device,
        "feature_count": predictor.feature_count,
        "load_time_ms": predictor.load_time_ms,
        "timestamp": datetime.utcnow().isoformat()
    }
