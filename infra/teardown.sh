#!/usr/bin/env bash
# Removes everything deploy.sh created (after judging ends).
set -euo pipefail
REGION="${AWS_REGION:-ap-southeast-2}"; export AWS_DEFAULT_REGION="$REGION"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
BUCKET_NAME="rustreader-scans-${ACCOUNT_ID}-${REGION}"
aws lambda delete-function --function-name rustreader-deep-scan || true
aws cloudwatch delete-dashboards --dashboard-names RustReader || true
aws s3 rm "s3://${BUCKET_NAME}" --recursive || true
aws s3api delete-bucket --bucket "$BUCKET_NAME" || true
aws iam delete-role-policy --role-name rustreader-deep-scan-role --policy-name scan-bucket-access || true
aws iam detach-role-policy --role-name rustreader-deep-scan-role --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole || true
aws iam delete-role --role-name rustreader-deep-scan-role || true
echo "Removed."
