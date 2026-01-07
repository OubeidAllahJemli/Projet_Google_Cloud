#!/bin/bash

# Movie Recommendation API - Deployment Script for GCP Cloud Run
# This script builds and deploys your Item-Based CF model to Cloud Run

set -e  # Exit on error

# ============================================================================
# CONFIGURATION - UPDATE THESE VALUES IF NEEDED
# ============================================================================

PROJECT_ID="students-group2"
REGION="europe-west1"
SERVICE_NAME="movie-recommender-api"
REPOSITORY="movie-recommender-repo"
IMAGE_NAME="movie-recommender"

# Full image path for Artifact Registry
IMAGE_PATH="$REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_NAME"

echo "======================================================================"
echo "DEPLOYING MOVIE RECOMMENDATION API TO GOOGLE CLOUD RUN"
echo "======================================================================"
echo ""
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo "Service Name: $SERVICE_NAME"
echo "Repository: $REPOSITORY"
echo "Image: $IMAGE_NAME"
echo ""

# ============================================================================
# STEP 1: Set the active project
# ============================================================================

echo "Step 1: Setting active GCP project..."
gcloud config set project $PROJECT_ID
echo "✓ Project set to: $PROJECT_ID"
echo ""

# ============================================================================
# STEP 2: Enable required APIs
# ============================================================================

echo "Step 2: Enabling required GCP APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com
echo "✓ APIs enabled"
echo ""

# ============================================================================
# STEP 3: Create Artifact Registry repository (if it doesn't exist)
# ============================================================================

echo "Step 3: Checking Artifact Registry repository..."

if gcloud artifacts repositories describe $REPOSITORY --location=$REGION &>/dev/null; then
    echo "✓ Repository '$REPOSITORY' already exists"
else
    echo "Creating Artifact Registry repository '$REPOSITORY'..."
    gcloud artifacts repositories create $REPOSITORY \
        --repository-format=docker \
        --location=$REGION \
        --description="Movie Recommender Docker images"
    echo "✓ Repository created successfully"
fi
echo ""

# ============================================================================
# STEP 4: Build the Docker image using Cloud Build
# ============================================================================

echo "Step 4: Building Docker image..."
echo "This may take 10-20 minutes due to large model file (814 MB)..."
echo ""

gcloud builds submit --tag $IMAGE_PATH

echo ""
echo "✓ Docker image built successfully"
echo ""

# ============================================================================
# STEP 5: Deploy to Cloud Run
# ============================================================================

echo "Step 5: Deploying to Cloud Run..."
echo ""

gcloud run deploy $SERVICE_NAME \
  --image $IMAGE_PATH \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 10 \
  --min-instances 0

echo ""
echo "✓ Deployment complete!"
echo ""

# ============================================================================
# STEP 6: Get the service URL
# ============================================================================

echo "Step 6: Getting service URL..."
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)')

echo ""
echo "======================================================================"
echo "✅ DEPLOYMENT SUCCESSFUL!"
echo "======================================================================"
echo ""
echo "🌐 Service URL: $SERVICE_URL"
echo ""
echo "📝 Test your API:"
echo ""
echo "   Health check:"
echo "   curl $SERVICE_URL/health"
echo ""
echo "   Get recommendations:"
echo "   curl -X POST $SERVICE_URL/recommend \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"user_ratings\": {\"1\": 5.0, \"50\": 4.5}, \"n_recommendations\": 5}'"
echo ""
echo "======================================================================"
