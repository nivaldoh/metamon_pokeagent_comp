#!/usr/bin/env bash
# Example Cloud Run deployment script for Metamon self-play battles
#
# Prerequisites:
# 1. Build and push the Docker image using cloudbuild.yaml
# 2. Set PROJECT_ID, REGION, and IMAGE_TAG variables below
# 3. Ensure you have appropriate Cloud Run permissions
# 4. (Optional) Create HuggingFace token and store in Secret Manager to avoid rate limits
#
# HuggingFace Authentication (optional but recommended):
#   To avoid API rate limits when downloading models, provide your HF token:
#
#   Option 1 - Using Google Secret Manager (recommended for production):
#     gcloud secrets create huggingface-token --data-file=- <<< "hf_xxxxxxxxxxxxx"
#     Then use: --set-secrets HF_TOKEN=huggingface-token:latest
#
#   Option 2 - Direct environment variable (simpler for testing):
#     --set-env-vars HF_TOKEN=hf_xxxxxxxxxxxxx
#
#   Get your token at: https://huggingface.co/settings/tokens

set -Eeuo pipefail

PROJECT_ID="${PROJECT_ID:-your-gcp-project-id}"
REGION="${REGION:-us-central1}"
REPO="${REPO:-metamon}"
IMAGE="${IMAGE:-metamon-sp}"
TAG="${TAG:-cpu}"
SERVICE_NAME="${SERVICE_NAME:-metamon-selfplay}"

IMAGE_URL="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}:${TAG}"

echo "Deploying Cloud Run service: ${SERVICE_NAME}"
echo "Using image: ${IMAGE_URL}"

# Example 1: Deploy with CLI arguments (recommended for simplicity)
# This overrides the container's default CMD
# Note: Add --set-secrets HF_TOKEN=huggingface-token:latest to avoid rate limits
gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE_URL}" \
  --region "${REGION}" \
  --platform managed \
  --memory 4Gi \
  --cpu 2 \
  --timeout 3600 \
  --max-instances 5 \
  --no-allow-unauthenticated \
  --args="--agent1=SyntheticRLV1,--agent2=SyntheticRLV2,--battle_format=gen1ou,--num_battles=1000"

# Example 1b: Same as above, but with HuggingFace authentication using Secret Manager
# gcloud run deploy "${SERVICE_NAME}" \
#   --image "${IMAGE_URL}" \
#   --region "${REGION}" \
#   --platform managed \
#   --memory 4Gi \
#   --cpu 2 \
#   --timeout 3600 \
#   --max-instances 5 \
#   --no-allow-unauthenticated \
#   --set-secrets HF_TOKEN=huggingface-token:latest \
#   --args="--agent1=SyntheticRLV1,--agent2=SyntheticRLV2,--battle_format=gen1ou,--num_battles=1000"

# Example 2: Deploy with environment variables
# Useful if you prefer using Cloud Run's UI for configuration
# Note: This example also shows passing HF_TOKEN directly (not recommended for production)
# gcloud run deploy "${SERVICE_NAME}" \
#   --image "${IMAGE_URL}" \
#   --region "${REGION}" \
#   --platform managed \
#   --memory 4Gi \
#   --cpu 2 \
#   --timeout 3600 \
#   --max-instances 5 \
#   --no-allow-unauthenticated \
#   --set-env-vars \
#     AGENT1=SyntheticRLV1,\
#     AGENT2=SyntheticRLV2,\
#     BATTLE_FORMAT=gen1ou,\
#     NUM_BATTLES=1000,\
#     CONCURRENCY=2,\
#     HF_TOKEN=hf_xxxxxxxxxxxxx

# Example 3: Deploy with trajectory saving to Cloud Storage
# Make sure the service account has write access to the bucket
# gcloud run deploy "${SERVICE_NAME}" \
#   --image "${IMAGE_URL}" \
#   --region "${REGION}" \
#   --platform managed \
#   --memory 8Gi \
#   --cpu 4 \
#   --timeout 3600 \
#   --max-instances 5 \
#   --no-allow-unauthenticated \
#   --set-env-vars \
#     AGENT1=SyntheticRLV1,\
#     AGENT2=SyntheticRLV2,\
#     BATTLE_FORMAT=gen1ou,\
#     NUM_BATTLES=5000,\
#     CONCURRENCY=4,\
#     SAVE_TRAJECTORIES_TO=/cache/trajectories

# Example 4: Generate training data with higher concurrency
# gcloud run deploy "${SERVICE_NAME}-data-gen" \
#   --image "${IMAGE_URL}" \
#   --region "${REGION}" \
#   --platform managed \
#   --memory 8Gi \
#   --cpu 4 \
#   --timeout 3600 \
#   --max-instances 10 \
#   --no-allow-unauthenticated \
#   --args="--agent1=SyntheticRLV2,--agent2=SyntheticRLV2,--battle_format=gen2ou,--num_battles=10000,--save_trajectories_to=/cache/gen2_selfplay"

echo ""
echo "Deployment complete!"
echo "To trigger the service (if it has auth):"
echo "  gcloud run services proxy ${SERVICE_NAME} --region ${REGION}"
echo ""
echo "To check logs:"
echo "  gcloud logging read 'resource.type=cloud_run_revision AND resource.labels.service_name=${SERVICE_NAME}' --limit 50 --format json"
