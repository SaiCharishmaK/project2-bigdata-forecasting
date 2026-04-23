# src/upload_to_s3.py
import boto3
import os
from dotenv import load_dotenv

load_dotenv()

def upload_raw_data():
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
    prefix = os.getenv(
        'S3_RAW_PREFIX', 'project2/raw/')
    raw_dir = 'data/raw'

    print(f"Uploading raw data to S3...")
    print(f"Bucket: {bucket}")
    print(f"Prefix: {prefix}\n")

    for filename in os.listdir(raw_dir):
        if filename.endswith('.parquet'):
            local_path = os.path.join(
                raw_dir, filename)
            s3_key = f"{prefix}{filename}"

            size_mb = os.path.getsize(
                local_path) / (1024*1024)
            print(f"Uploading {filename} "
                  f"({size_mb:.1f} MB)...")

            s3.upload_file(
                local_path, bucket, s3_key)
            print(f"  ✅ s3://{bucket}/{s3_key}")

    print("\n✅ All files uploaded to S3")

if __name__ == "__main__":
    upload_raw_data()