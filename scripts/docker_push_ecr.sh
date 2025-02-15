#!/bin/sh
# This script pushes a Docker image to AWS ECR.
# Usage: sudo -E sh docker_push_ecr.sh --name=<image_name> --tag=<docker_tag> --region=<aws_region> --account=<aws_account_id>

# Function to display usage
usage() {
    echo "Usage: $0 --name=<image_name> --tag=<docker_tag> --region=<aws_region> --account=<aws_account_id>"
    exit 1
}

# Parse arguments
while [ "$#" -gt 0 ]; do
    case $1 in
        --name=*)
            IMAGE_NAME="${1#*=}"
            ;;
        --tag=*)
            TAG="${1#*=}"
            ;;
        --region=*)
            AWS_REGION="${1#*=}"
            ;;
        --account=*)
            AWS_ACCOUNT_ID="${1#*=}"
            ;;
        *)
            echo "Error: Invalid argument: $1"
            usage
            ;;
    esac
    shift
done

# Ensure required parameters are provided
if [ -z "$IMAGE_NAME" ] || [ -z "$TAG" ] || [ -z "$AWS_REGION" ] || [ -z "$AWS_ACCOUNT_ID" ]; then
    echo "Error: Missing required parameters (--name, --tag, --region, or --account)."
    usage
fi

# Construct the ECR repository URI
ECR_ID="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
echo "Pushing image to ECR: ${ECR_ID}/${IMAGE_NAME}:${TAG}"

# Authenticate Docker with AWS ECR
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "${ECR_ID}"

# Tag and push the image
docker tag "${IMAGE_NAME}:${TAG}" "${ECR_ID}/${IMAGE_NAME}:${TAG}"
docker push "${ECR_ID}/${IMAGE_NAME}:${TAG}"
