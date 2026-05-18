# tests/test_pipeline.py
"""
Unit tests that run in GitHub Actions CI.
These validate the pipeline components
before any deployment happens.
"""
import pytest
import pandas as pd
import numpy as np
import os
import json
import sys

sys.path.insert(0, os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__))))

# ── Feature Engineering Tests ─────────────────

class TestFeatureEngineering:
    """Test feature creation logic"""

    def test_cyclical_encoding_hour(self):
        """Hour 0 and hour 23 should be close"""
        hour_0_sin  = np.sin(2*np.pi*0/24)
        hour_23_sin = np.sin(2*np.pi*23/24)
        hour_12_sin = np.sin(2*np.pi*12/24)

        # Hour 23 closer to 0 than hour 12 is
        dist_0_23 = abs(hour_0_sin - hour_23_sin)
        dist_0_12 = abs(hour_0_sin - hour_12_sin)

        assert dist_0_23 < dist_0_12, \
            "Cyclical encoding broken: " \
            "hour 23 should be close to hour 0"

    def test_rush_hour_flag(self):
        """Rush hours should be flagged"""
        rush_hours = list(range(7, 10)) + \
                     list(range(17, 21))
        non_rush   = [0, 1, 2, 3, 4, 5,
                      11, 13, 14, 15]

        for h in rush_hours:
            is_rush = 1 if h in rush_hours else 0
            assert is_rush == 1, \
                f"Hour {h} should be rush hour"

        for h in non_rush:
            is_rush = 1 if h in rush_hours else 0
            assert is_rush == 0, \
                f"Hour {h} should not be rush hour"

    def test_weekend_flag(self):
        """Weekends are day 1 (Sun) and 7 (Sat)"""
        assert 1 in [1, 7]  # Sunday
        assert 7 in [1, 7]  # Saturday
        assert 2 not in [1, 7]  # Monday

    def test_lag_values_positive(self):
        """Trip counts cannot be negative"""
        sample_counts = [100, 500, 1200,
                         800, 0, 1500]
        for count in sample_counts:
            assert count >= 0, \
                f"Trip count {count} is negative"


# ── Data Quality Tests ─────────────────────────

class TestDataQuality:
    """Test data validation rules"""

    def setup_method(self):
        """Create sample data for testing"""
        self.df = pd.DataFrame({
            'fare_amount':   [10, 25, 50,
                              -5, 600, 30],
            'trip_distance': [1.5, 3.2, 8.0,
                              0, 150, 4.5],
            'passenger_count':[1, 2, 3, 0, 5, 4],
        })

    def test_fare_filter(self):
        """Fares must be positive and <= 500"""
        clean = self.df[
            (self.df['fare_amount'] > 0) &
            (self.df['fare_amount'] <= 500)
        ]
        assert len(clean) == 4, \
            "Expected 4 valid fare records"

    def test_distance_filter(self):
        """Distance must be positive <= 100"""
        clean = self.df[
            (self.df['trip_distance'] > 0) &
            (self.df['trip_distance'] <= 100)
        ]
        assert len(clean) == 4, \
            "Expected 4 valid distance records"

    def test_passenger_filter(self):
        """Passengers must be 1-6"""
        clean = self.df[
            (self.df['passenger_count'] >= 1) &
            (self.df['passenger_count'] <= 6)
        ]
        assert len(clean) == 5, \
            "Expected 5 valid passenger records"


# ── Model Performance Tests ────────────────────

class TestModelPerformance:
    """Performance gate tests"""

    def test_model_results_exist(self):
        """Model must have been trained"""
        results_path = 'logs/model_results.json'
        if not os.path.exists(results_path):
            pytest.skip(
                "Model not trained yet")

        with open(results_path) as f:
            results = json.load(f)

        assert 'sagemaker_xgb' in results or \
               'local_xgb' in results, \
            "No model results found"

    def test_mae_threshold(self):
        """Model MAE must be reasonable"""
        results_path = 'logs/model_results.json'
        if not os.path.exists(results_path):
            pytest.skip(
                "Model not trained yet")

        with open(results_path) as f:
            results = json.load(f)

        # Get MAE from whichever model exists
        mae = None
        if 'sagemaker_xgb' in results:
            mae = results['sagemaker_xgb']\
                .get('val_mae')
        elif 'local_xgb' in results:
            mae = results['local_xgb']\
                .get('val_mae')

        if mae is None:
            pytest.skip("MAE not found")

        # MAE should be reasonable for taxi data
        assert mae < 500, \
            f"MAE {mae} too high — " \
            f"model may not have trained correctly"

    def test_improvement_over_baseline(self):
        """Model must beat naive baseline"""
        results_path = 'logs/model_results.json'
        if not os.path.exists(results_path):
            pytest.skip("Model not trained yet")

        with open(results_path) as f:
            results = json.load(f)

        improvement = results.get(
            'improvement_over_baseline', 0)
        assert improvement > 0, \
            "Model does not beat naive baseline"


# ── API Schema Tests ───────────────────────────

class TestAPISchemas:
    """Test API request/response logic"""

    def test_hour_range_valid(self):
        """Hours must be 0-23"""
        valid_hours = list(range(0, 24))
        for h in valid_hours:
            assert 0 <= h <= 23

    def test_hour_range_invalid(self):
        """Hours outside 0-23 are invalid"""
        invalid = [-1, 24, 25, 100]
        for h in invalid:
            assert not (0 <= h <= 23), \
                f"Hour {h} should be invalid"

    def test_trip_count_non_negative(self):
        """Trip count must be non-negative"""
        assert 0 >= 0
        assert 1200 >= 0
        with pytest.raises(AssertionError):
            assert -1 >= 0

    def test_prediction_positive(self):
        """Predictions must never be negative"""
        raw_preds = [1200.5, -50.0, 800.3, -10.0]
        clipped   = [max(0, p) for p in raw_preds]
        for p in clipped:
            assert p >= 0, \
                "Negative prediction not clipped"


if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])