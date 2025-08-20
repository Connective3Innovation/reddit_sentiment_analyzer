#!/usr/bin/env bash
set -euo pipefail

# ── 0. CONFIG ─────────────────────────────────────────────────────
PROJECT="c3datahub-308511"
REGION="europe-west2"

REPO="reddit-sentiment"
IMAGE="$REGION-docker.pkg.dev/$PROJECT/$REPO/api:latest"
SERVICE="reddit-sentiment-api"
SA="reddit-sentiment-sa"

# Bucket for outputs and cache
BUCKET="${PROJECT}-reddit-sentiment"
GCS_PREFIX="gs://${BUCKET}"

# Secret names
SEC_REDDIT_CLIENT_ID="REDDIT_CLIENT_ID"
SEC_REDDIT_CLIENT_SECRET="REDDIT_CLIENT_SECRET"
SEC_REDDIT_USER_AGENT="REDDIT_USER_AGENT"

# >>>> FILL THESE IN (or export before running) <<<<
VAL_REDDIT_CLIENT_ID="Z7YwxnsoMO8vAqYsttwgGg"
VAL_REDDIT_CLIENT_SECRET="F2pPHmltdgOp4yUSpfWtwW_qiuigBQ"
VAL_REDDIT_USER_AGENT="sentiment-project (by Gaggan Bajwwa)"

# ── 1. Set active project ─────────────────────────────────────────
gcloud config set project "$PROJECT"

# ── 2. Enable required APIs ───────────────────────────────────────
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  secretmanager.googleapis.com \
  storage.googleapis.com

# ── 3. Create Artifact Registry repo (idempotent) ─────────────────
gcloud artifacts repositories create "$REPO" \
  --repository-format=docker \
  --location="$REGION" \
  --description="Docker images for Reddit Sentiment" || echo "Repo exists"

# ── 4. Create / ensure runtime service account ────────────────────
gcloud iam service-accounts create "$SA" \
  --display-name="Reddit Sentiment runtime" || echo "SA exists"

# Grant runtime permissions
gcloud projects add-iam-policy-binding "$PROJECT" \
  --member="serviceAccount:${SA}@${PROJECT}.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"

gcloud projects add-iam-policy-binding "$PROJECT" \
  --member="serviceAccount:${SA}@${PROJECT}.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

gcloud projects add-iam-policy-binding "$PROJECT" \
  --member="serviceAccount:${SA}@${PROJECT}.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.reader"

# ── 5. Secret Manager: create or update Reddit creds ──────────────
echo -n "$VAL_REDDIT_CLIENT_ID"    > clientid.txt
echo -n "$VAL_REDDIT_CLIENT_SECRET" > clientsecret.txt
echo -n "$VAL_REDDIT_USER_AGENT"   > useragent.txt

gcloud secrets create "$SEC_REDDIT_CLIENT_ID" --data-file=clientid.txt 2>/dev/null \
  || gcloud secrets versions add "$SEC_REDDIT_CLIENT_ID" --data-file=clientid.txt

gcloud secrets create "$SEC_REDDIT_CLIENT_SECRET" --data-file=clientsecret.txt 2>/dev/null \
  || gcloud secrets versions add "$SEC_REDDIT_CLIENT_SECRET" --data-file=clientsecret.txt

gcloud secrets create "$SEC_REDDIT_USER_AGENT" --data-file=useragent.txt 2>/dev/null \
  || gcloud secrets versions add "$SEC_REDDIT_USER_AGENT" --data-file=useragent.txt

rm -f clientid.txt clientsecret.txt useragent.txt

# ── 6. GCS bucket for cache + outputs ─────────────────────────────
gsutil mb -l "$REGION" "gs://$BUCKET" 2>/dev/null || echo "Bucket exists"

# ── 7. Build & push Docker image via Cloud Build ──────────────────
gcloud builds submit --tag "$IMAGE" .

# ── 8. Deploy to Cloud Run ────────────────────────────────────────
gcloud run deploy "$SERVICE" \
  --image "$IMAGE" \
  --region "$REGION" \
  --service-account "${SA}@${PROJECT}.iam.gserviceaccount.com" \
  --cpu 2 \
  --memory 8Gi \
  --concurrency=10 \
  --min-instances=1 \
  --timeout=900s \
  --port 8080 \
  --max-instances=3 \
  --allow-unauthenticated \
  --set-env-vars \
    "REDDIT_OUTPUT_DIR=${GCS_PREFIX}/etl,REDDIT_CACHE_BACKEND=gcs,REDDIT_GCS_CACHE_PREFIX=${GCS_PREFIX}/cache,REDDIT_GCP_PROJECT=${PROJECT},REDDIT_CACHE_MAX_AGE_HOURS=24,REDDIT_MAX_POSTS=500,REDDIT_SENTIMENT_ENGINE=vader,REDDIT_MORE_LIMIT=2,REDDIT_COMMENTS_PER_POST=300" \
  --set-secrets \
    "REDDIT_CLIENT_ID=${SEC_REDDIT_CLIENT_ID}:latest,REDDIT_CLIENT_SECRET=${SEC_REDDIT_CLIENT_SECRET}:latest,REDDIT_USER_AGENT=${SEC_REDDIT_USER_AGENT}:latest"

# ── 9. Show service URL ──────────────────────────────────────────
SERVICE_URL=$(gcloud run services describe "$SERVICE" --region "$REGION" --format="value(status.url)")
echo "Service URL: $SERVICE_URL"
