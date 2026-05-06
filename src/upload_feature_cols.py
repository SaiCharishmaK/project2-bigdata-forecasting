# Run this locally in a Python file
# src/upload_feature_cols.py
import boto3
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Load feature cols from local results
with open('logs/local_model_results.json') as f:
    results = json.load(f)

feature_cols = results['feature_cols']

s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv(
        'AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv(
        'AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_DEFAULT_REGION')
)

bucket = os.getenv('S3_BUCKET_DATA')

# Upload feature cols
s3.put_object(
    Bucket=bucket,
    Key='project2/features/feature_cols.json',
    Body=json.dumps(feature_cols),
    ContentType='application/json'
)
print(f"✅ Uploaded feature_cols.json to S3")

# Also upload the local model
s3.upload_file(
    'models/xgb_champion.json',
    bucket,
    'project2/sagemaker/xgb_model.json'
)
print(f"✅ Uploaded xgb_model.json to S3")