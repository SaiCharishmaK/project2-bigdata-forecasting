# src/train_sagemaker.py
"""
SageMaker training script.
SageMaker runs this file on a managed instance.
Data is downloaded from S3 automatically.
Model artifact saved to /opt/ml/model/
"""
import os
import json
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

def parse_args():
    parser = argparse.ArgumentParser()

    # SageMaker hyperparameters
    parser.add_argument(
        '--n-estimators', type=int,
        default=500)
    parser.add_argument(
        '--max-depth', type=int, default=6)
    parser.add_argument(
        '--learning-rate', type=float,
        default=0.05)
    parser.add_argument(
        '--subsample', type=float,
        default=0.8)
    parser.add_argument(
        '--colsample-bytree', type=float,
        default=0.8)

    # SageMaker environment variables
    # These are set automatically by SageMaker
    parser.add_argument(
        '--model-dir',
        type=str,
        default=os.environ.get(
            'SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument(
        '--train',
        type=str,
        default=os.environ.get(
            'SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
    parser.add_argument(
        '--validation',
        type=str,
        default=os.environ.get(
            'SM_CHANNEL_VALIDATION',
            '/opt/ml/input/data/validation'))

    return parser.parse_args()

def load_data(data_dir):
    """Load CSV data from SageMaker channel"""
    files = [
        f for f in os.listdir(data_dir)
        if f.endswith('.csv')
    ]
    dfs = [
        pd.read_csv(os.path.join(data_dir, f))
        for f in files
    ]
    return pd.concat(dfs, ignore_index=True)

def get_feature_cols(df):
    """Get feature columns excluding target"""
    exclude = [
        'pickup_hour', 'pickup_date',
        'trip_count',
        'target_trip_count_1h',
        'target_trip_count_6h',
        'target_trip_count_24h'
    ]
    return [
        c for c in df.columns
        if c not in exclude
    ]

def train(args):
    print("Loading training data...")
    df_train = load_data(args.train)
    df_val   = load_data(args.validation)

    feature_cols = get_feature_cols(df_train)
    target       = 'target_trip_count_24h'

    X_train = df_train[feature_cols].fillna(0)
    y_train = df_train[target].fillna(0)
    X_val   = df_val[feature_cols].fillna(0)
    y_val   = df_val[target].fillna(0)

    print(f"Train: {X_train.shape}")
    print(f"Val:   {X_val.shape}")
    print(f"Features: {len(feature_cols)}")

    # Train XGBoost
    print("\nTraining XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
        eval_metric='mae'
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )

    # Evaluate
    val_pred = model.predict(X_val)
    mae  = mean_absolute_error(y_val, val_pred)
    rmse = np.sqrt(mean_squared_error(
        y_val, val_pred))
    mape = np.mean(
        np.abs((y_val - val_pred) /
               (y_val + 1e-8)) * 100)
    r2   = r2_score(y_val, val_pred)

    print(f"\n=== SAGEMAKER TRAINING RESULTS ===")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R2:   {r2:.4f}")

    # Save model to SageMaker model dir
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(
        args.model_dir, 'xgb_model.json')
    model.save_model(model_path)

    # Save metrics alongside model
    metrics = {
        'val_mae':  round(mae, 4),
        'val_rmse': round(rmse, 4),
        'val_mape': round(mape, 4),
        'val_r2':   round(r2, 4),
        'best_iteration':
            int(model.best_iteration),
        'feature_cols': feature_cols,
        'hyperparameters': {
            'n_estimators': args.n_estimators,
            'max_depth':    args.max_depth,
            'learning_rate':args.learning_rate,
        }
    }

    metrics_path = os.path.join(
        args.model_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"\n✅ Model saved to: {model_path}")
    print(f"✅ Metrics saved to: {metrics_path}")

    return metrics

if __name__ == '__main__':
    args   = parse_args()
    result = train(args)
    print("\nSageMaker training complete!")
    print(json.dumps(result, indent=4))