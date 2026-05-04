#!/usr/bin/env python3
"""
Add iam:PassRole policy to ml-portfolio-user for SageMaker training.
Run this to fix the "User is not authorized to perform: iam:PassRole" error.
"""

import boto3
import json
import os
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv()

# Create IAM client using credentials from .env
iam_client = boto3.client(
    'iam',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
)

policy_document = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": "iam:PassRole",
            "Resource": "arn:aws:iam::004285426490:role/SageMakerExecutionRole"
        }
    ]
}

try:
    print("Adding iam:PassRole policy to ml-portfolio-user...")
    iam_client.put_user_policy(
        UserName='ml-portfolio-user',
        PolicyName='SageMakerPassRole',
        PolicyDocument=json.dumps(policy_document)
    )
    print("✅ Policy added successfully!")
    print("\nVerifying policy was applied...")
    
    # Verify the policy
    response = iam_client.list_user_policies(UserName='ml-portfolio-user')
    if 'SageMakerPassRole' in response['PolicyNames']:
        print("✅ SageMakerPassRole policy is now active!")
    else:
        print("⚠️  Policy not found in list. This may take a few seconds to propagate.")
        
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nPossible fixes:")
    print("1. Ensure your AWS credentials are configured: aws configure")
    print("2. Ensure you have IAM permissions to manage policies")
    print("3. Or skip SageMaker and use local training instead")
