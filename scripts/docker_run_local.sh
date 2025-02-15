#!/bin/sh
# This script runs a Docker container.
# Usage: sudo sh docker_build.sh --name=<image_name> --tag=<docker_tag> [--port=<port>]

# Function to display usage
usage() {
    echo "Usage: $0 --name=<image_name> --tag=<docker_tag> [--port=<port>]"
    exit 1
}

# Set default values
PORT=8080

# Parse arguments
while [ "$#" -gt 0 ]; do
    case $1 in
        --name=*)
            IMAGE_NAME="${1#*=}"
            ;;
        --tag=*)
            TAG="${1#*=}"
            ;;
        --port=*)
            PORT="${1#*=}"
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

# Run the Docker container
echo "Starting Docker container: ${IMAGE_NAME} with image ${IMAGE_NAME}:${TAG} on port ${PORT}"
docker run -p "8080:${PORT}" --name="$IMAGE_NAME" --shm-size=32000m --rm "$IMAGE_NAME:$TAG"
