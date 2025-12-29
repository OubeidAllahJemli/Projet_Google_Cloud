#!/bin/bash

# Movie Recommendation API - Simple Source-Based Deployment
set -e

# Configuration
PROJECT_ID="students-group2"
REGION="europe-west1"
SERVICE_NAME="movie-recommender-api"

echo "======================================================================"
echo "DEPLOYING TO GOOGLE CLOUD RUN (Source-Based)"
echo "======================================================================"
echo ""
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Service: $SERVICE_NAME"
echo ""
echo "⚠️  This will take 15-20 minutes due to large model file (814 MB)"
echo ""

# Set project
echo "Setting project..."
gcloud config set project $PROJECT_ID
echo "✓ Project set"
echo ""

# Deploy directly from source
echo "Building and deploying..."
echo "Progress will show below:"
echo ""

gcloud run deploy $SERVICE_NAME \
  --source . \
  --region=$REGION \
  --allow-unauthenticated \
  --memory=2Gi \
  --cpu=2 \
  --timeout=300 \
  --max-instances=10 \
  --min-instances=0 \
  --platform=managed

# Get service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)')

echo ""
echo "======================================================================"
echo "✅ DEPLOYMENT SUCCESSFUL!"
echo "======================================================================"
echo ""
echo "🌐 Your API is live at:"
echo "   $SERVICE_URL"
echo ""
echo "📝 Test it:"
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
