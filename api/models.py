# api/models.py
"""
Pydantic models for taxi demand forecasting API.
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

# ── Request Models ─────────────────────────────

class ForecastRequest(BaseModel):
    """Single forecast request"""
    features: List[float] = Field(
        ...,
        min_items=31,
        max_items=31,
        description="31 features for 24-hour forecast"
    )
    location: Optional[str] = Field(
        default="unknown",
        description="Location/zone identifier"
    )

class BatchForecastRequest(BaseModel):
    """Batch forecast request"""
    forecasts: List[ForecastRequest] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="Batch of forecast requests"
    )

# ── Response Models ────────────────────────────

class ForecastResponse(BaseModel):
    """Single forecast response"""
    predicted_trip_count: float = Field(
        ...,
        description="Predicted 24-hour trip count"
    )
    prediction_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence (0-1)"
    )
    location: str
    forecast_horizon: str = "24 hours"
    model_version: str
    processing_time_ms: float
    timestamp: str

class BatchForecastResponse(BaseModel):
    """Batch forecast response"""
    total_forecasts: int
    successful: int
    failed: int
    predictions: List[ForecastResponse]
    batch_processing_time_ms: float
    timestamp: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_name: str
    model_version: str
    device: str
    timestamp: str
    feature_count: int
