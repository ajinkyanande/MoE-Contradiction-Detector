#!/bin/sh
# This script builds a Docker image.
# Usage: sudo sh docker_build.sh --name=<image_name> --tag=<docker_tag>

# Function to display usage
usage() {
    echo "Usage: $0 --name=<image_name> --tag=<docker_tag>"
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
        *)
            echo "Error: Invalid argument: $1"
            usage
            ;;
    esac
    shift
done

# Ensure required parameters are provided
if [ -z "$IMAGE_NAME" ] || [ -z "$TAG" ]; then
    echo "Error: Missing required parameters (--name or --tag)."
    usage
fi

# Build the Docker image
echo "Building Docker image: ${IMAGE_NAME}:${TAG}"
docker build --platform linux/amd64 -t "${IMAGE_NAME}:${TAG}" .
