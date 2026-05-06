# api/predictor.py
"""
Model loading and inference for taxi demand forecasting.
XGBoost model loaded ONCE at startup.
"""
import xgboost as xgb
import numpy as np
import time
import os
import json
from dotenv import load_dotenv

load_dotenv()

class ForecastPredictor:
    """
    Handles XGBoost model inference.
    Single instance shared across all API requests.
    """

    def __init__(self):
        self.model = None
        self.model_version = "1.0.0"
        self.is_loaded = False
        self.feature_count = 31
        self.device = "cpu"
        self.load_time_ms = 0

    def load_models(self):
        """
        Load XGBoost model from local file.
        Called once at API startup.
        """
        start_time = time.time()
        
        try:
            # Try to load from local directory first
            model_path = 'models/xgb_champion.json'
            if not os.path.exists(model_path):
                # Fallback to SageMaker model
                model_path = \
                    'models/sagemaker/xgb_model.json'
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model not found at {model_path}")
            
            self.model = xgb.XGBRegressor()
            self.model.load_model(model_path)
            
            self.is_loaded = True
            self.load_time_ms = \
                (time.time() - start_time) * 1000
            
            print(f"✅ Model loaded from {model_path}")
            print(f"   Load time: "
                  f"{self.load_time_ms:.1f}ms")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self.is_loaded = False

    def predict(self, features: np.ndarray) \
            -> tuple:
        """
        Make prediction on input features.
        
        Args:
            features: Array of shape (31,) or (n, 31)
        
        Returns:
            (prediction, confidence)
        """
        if not self.is_loaded:
            raise RuntimeError(
                "Model not loaded. "
                "Call load_models() first.")
        
        # Ensure 2D input
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Make prediction
        prediction = self.model.predict(features)
        
        # Get base prediction and confidence
        # (XGBoost doesn't have native confidence,
        # so we use prediction variance as proxy)
        if prediction.ndim > 0:
            pred = prediction[0]
        else:
            pred = prediction
        
        # Simple confidence: assume higher 
        # predictions are more confident
        # (In production, use Bayesian methods)
        confidence = min(1.0, 
                        abs(float(pred)) / 1000.0)
        
        return float(pred), confidence

    def validate_features(self, 
                         features: list) -> bool:
        """Validate input feature count"""
        return len(features) == self.feature_count


# Singleton predictor instance
predictor = ForecastPredictor()
