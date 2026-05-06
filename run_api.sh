#!/bin/bash
# run_api.sh - Start the Taxi Demand Forecasting API

echo "🚀 Starting Taxi Demand Forecasting API..."
echo ""

# Check if running in venv
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: Not in a virtual environment"
    echo "   Consider activating: source venv/bin/activate"
fi

# Check if model exists
if [ ! -f "models/xgb_champion.json" ] && \
   [ ! -f "models/sagemaker/xgb_model.json" ]; then
    echo "❌ Error: Model not found!"
    echo "   Expected: models/xgb_champion.json"
    echo "   Or: models/sagemaker/xgb_model.json"
    exit 1
fi

echo "✅ Model found"
echo "📦 Starting API server on 0.0.0.0:8000"
echo ""
echo "📚 Documentation: http://localhost:8000/docs"
echo "🏥 Health check: http://localhost:8000/health"
echo ""

# Run with uvicorn
python -m uvicorn api.app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload
