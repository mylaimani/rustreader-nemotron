#!/usr/bin/env bash
# Deploys RustReader Deep Scan to AWS. Run from the rustreader_cloud folder.
# Easiest place to run it: AWS CloudShell (AWS CLI is already signed in there).
#
#   bash infra/deploy.sh                 # region defaults to ap-southeast-2 (Sydney)
#   AWS_REGION=ap-southeast-2 bash infra/deploy.sh   (CloudShell sets AWS_REGION; pass it explicitly)
#
# Creates (or updates): S3 bucket, IAM role, Lambda (Python 3.12, arm64 Graviton),
# public Function URL, log retention and a CloudWatch dashboard.
set -euo pipefail

REGION="${AWS_REGION:-ap-southeast-2}"
# Override the function name to run a second copy side by side, e.g. the Nebius
# hackathon one:  RUSTREADER_FUNCTION_NAME=rustreader-nemotron bash infra/deploy.sh
FUNCTION_NAME="${RUSTREADER_FUNCTION_NAME:-rustreader-deep-scan}"
ROLE_NAME="rustreader-deep-scan-role"
# NEBIUS_API_KEY (optional): enables NVIDIA Nemotron on Nebius Token Factory.
# Export it in the shell before running; it is stored only as a Lambda env var.
if [ -z "${NEBIUS_API_KEY:-}" ]; then
  echo "NOTE: NEBIUS_API_KEY is not set -> deploying WITHOUT Nemotron (pure OpenCV mode)."
fi
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
BUCKET_NAME="rustreader-scans-${ACCOUNT_ID}-${REGION}"
BUILD_FOLDER="build_lambda"
export AWS_DEFAULT_REGION="$REGION"

echo "== 1/6 S3 bucket ${BUCKET_NAME} (private, scans auto-delete after 30 days)"
if ! aws s3api head-bucket --bucket "$BUCKET_NAME" 2>/dev/null; then
  if [ "$REGION" = "us-east-1" ]; then
    aws s3api create-bucket --bucket "$BUCKET_NAME" >/dev/null
  else
    aws s3api create-bucket --bucket "$BUCKET_NAME" --create-bucket-configuration "LocationConstraint=${REGION}" >/dev/null
  fi
fi
aws s3api put-public-access-block --bucket "$BUCKET_NAME" --public-access-block-configuration \
  "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
aws s3api put-bucket-lifecycle-configuration --bucket "$BUCKET_NAME" --lifecycle-configuration \
  '{"Rules":[{"ID":"expire-scans","Status":"Enabled","Filter":{"Prefix":"scans/"},"Expiration":{"Days":30}},
             {"ID":"expire-sessions","Status":"Enabled","Filter":{"Prefix":"sessions/"},"Expiration":{"Days":30}}]}'

echo "== 2/6 IAM role (logs + this bucket only)"
if ! aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
  aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document \
    '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
  aws iam attach-role-policy --role-name "$ROLE_NAME" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
  echo "   waiting 15 s for the new role to propagate"; sleep 15
fi
aws iam put-role-policy --role-name "$ROLE_NAME" --policy-name scan-bucket-access --policy-document \
  "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\",\"s3:PutObject\"],\"Resource\":\"arn:aws:s3:::${BUCKET_NAME}/*\"}]}"
ROLE_ARN="$(aws iam get-role --role-name "$ROLE_NAME" --query Role.Arn --output text)"

echo "== 3/6 Build Lambda bundle (OpenCV 5 wheels for arm64 / Python 3.12)"
rm -rf "$BUILD_FOLDER" && mkdir -p "$BUILD_FOLDER/package"
python3 -m pip install --quiet --platform manylinux2014_aarch64 --python-version 3.12 --implementation cp \
  --only-binary=:all: --target "$BUILD_FOLDER/package" -r infra/requirements-lambda.txt
cp -r vision agent api web "$BUILD_FOLDER/package/"
mkdir -p "$BUILD_FOLDER/package/docs" "$BUILD_FOLDER/package/samples/synthetic"
cp docs/RustReader_Measuring_Mat_A4.pdf "$BUILD_FOLDER/package/docs/"
cp samples/synthetic/M12x1.25x50_heavy.jpg samples/synthetic/fail_two_bolts.jpg samples/synthetic/fail_blurry.jpg \
   "$BUILD_FOLDER/package/samples/synthetic/"
find "$BUILD_FOLDER/package" -name "__pycache__" -type d -prune -exec rm -rf {} +
(cd "$BUILD_FOLDER/package" && zip -q -r ../function.zip .)
aws s3 cp "$BUILD_FOLDER/function.zip" "s3://${BUCKET_NAME}/deploy/function.zip" --only-show-errors

echo "== 4/6 Lambda function ${FUNCTION_NAME}"
if aws lambda get-function --function-name "$FUNCTION_NAME" >/dev/null 2>&1; then
  aws lambda update-function-code --function-name "$FUNCTION_NAME" \
    --s3-bucket "$BUCKET_NAME" --s3-key deploy/function.zip >/dev/null
  aws lambda wait function-updated --function-name "$FUNCTION_NAME"
  aws lambda update-function-configuration --function-name "$FUNCTION_NAME" \
    --environment "Variables={SCAN_BUCKET=${BUCKET_NAME},NEBIUS_API_KEY=${NEBIUS_API_KEY:-}}" \
    --memory-size 2048 --timeout 60 >/dev/null
else
  aws lambda create-function --function-name "$FUNCTION_NAME" --runtime python3.12 --architectures arm64 \
    --handler api.lambda_handler.lambda_handler --role "$ROLE_ARN" \
    --code "S3Bucket=${BUCKET_NAME},S3Key=deploy/function.zip" \
    --memory-size 2048 --timeout 60 \
    --environment "Variables={SCAN_BUCKET=${BUCKET_NAME},NEBIUS_API_KEY=${NEBIUS_API_KEY:-}}" >/dev/null
fi
aws lambda wait function-active-v2 --function-name "$FUNCTION_NAME"
aws lambda wait function-updated --function-name "$FUNCTION_NAME"

echo "== 5/6 Public Function URL"
if ! aws lambda get-function-url-config --function-name "$FUNCTION_NAME" >/dev/null 2>&1; then
  aws lambda create-function-url-config --function-name "$FUNCTION_NAME" --auth-type NONE >/dev/null
  aws lambda add-permission --function-name "$FUNCTION_NAME" --statement-id PublicUrlInvokeUrl \
    --action lambda:InvokeFunctionUrl --principal '*' --function-url-auth-type NONE >/dev/null
  aws lambda add-permission --function-name "$FUNCTION_NAME" --statement-id PublicUrlInvokeFunction \
    --action lambda:InvokeFunction --principal '*' --invoked-via-function-url >/dev/null
fi
FUNCTION_URL="$(aws lambda get-function-url-config --function-name "$FUNCTION_NAME" --query FunctionUrl --output text)"

echo "== 6/6 Logs retention + CloudWatch dashboard"
aws logs create-log-group --log-group-name "/aws/lambda/${FUNCTION_NAME}" 2>/dev/null || true
aws logs put-retention-policy --log-group-name "/aws/lambda/${FUNCTION_NAME}" --retention-in-days 14
sed -e "s/__REGION__/${REGION}/g" -e "s/__FUNCTION__/${FUNCTION_NAME}/g" infra/dashboard.json > "$BUILD_FOLDER/dashboard.json"
DASHBOARD_NAME="RustReader"
if [ "$FUNCTION_NAME" != "rustreader-deep-scan" ]; then DASHBOARD_NAME="RustReader-${FUNCTION_NAME#rustreader-}"; fi
aws cloudwatch put-dashboard --dashboard-name "$DASHBOARD_NAME" --dashboard-body "file://${BUILD_FOLDER}/dashboard.json" >/dev/null

echo
echo "Deployed. Open: ${FUNCTION_URL}"
echo "Smoke test:    bash infra/smoke_test.sh ${FUNCTION_URL}"
