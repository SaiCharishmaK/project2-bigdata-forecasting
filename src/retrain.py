# src/retrain.py
"""
Lightweight retraining script.
Used by GitHub Actions and manually.
"""
import os
import sys
import json
import boto3
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    mean_absolute_error, r2_score)
from dotenv import load_dotenv

load_dotenv()

def retrain():
    print("="*50)
    print("  RETRAINING PIPELINE")
    print("="*50)

    # Load data
    df_train = pd.read_csv(
        'data/features/train.csv')
    df_val   = pd.read_csv(
        'data/features/val.csv')

    with open(
        'data/features/feature_cols.json'
    ) as f:
        feature_cols = json.load(f)

    TARGET   = 'target_trip_count_24h'
    X_train  = df_train[feature_cols].fillna(0)
    y_train  = df_train[TARGET].fillna(0)
    X_val    = df_val[feature_cols].fillna(0)
    y_val    = df_val[TARGET].fillna(0)

    print(f"Train: {X_train.shape}")
    print(f"Val:   {X_val.shape}")

    # Train
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
        eval_metric='mae'
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )

    # Evaluate
    val_pred = model.predict(X_val)
    mae  = mean_absolute_error(y_val, val_pred)
    r2   = r2_score(y_val, val_pred)
    mape = np.mean(np.abs(
        (y_val - val_pred) /
        (y_val + 1e-8)) * 100)

    print(f"\n✅ Training complete")
    print(f"   MAE:  {mae:.2f}")
    print(f"   MAPE: {mape:.2f}%")
    print(f"   R2:   {r2:.4f}")

    # Performance gate
    if mae > 500:
        print(f"❌ MAE {mae:.2f} above threshold")
        sys.exit(1)

    # Save model
    os.makedirs('models', exist_ok=True)
    model.save_model('models/xgb_retrained.json')

    # Upload to S3
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv(
            'AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv(
            'AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv(
            'AWS_DEFAULT_REGION')
    )
    bucket = os.getenv('S3_BUCKET_DATA')

    s3.upload_file(
        'models/xgb_retrained.json',
        bucket,
        'project2/sagemaker/xgb_model.json'
    )

    results = {
        'val_mae':        round(mae, 4),
        'val_mape':       round(mape, 4),
        'val_r2':         round(r2, 4),
        'n_features':     len(feature_cols),
        'best_iteration': int(
            model.best_iteration),
    }

    os.makedirs('logs', exist_ok=True)
    with open('logs/retrain_results.json',
              'w') as f:
        json.dump(results, f, indent=4)

    print(f"✅ Model saved and uploaded to S3")
    print(json.dumps(results, indent=4))
    return results

if __name__ == "__main__":
    retrain()