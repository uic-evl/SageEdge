#!/bin/bash

# Test server script for NIU Image Stream Server
# This script runs the server using Docker

echo "Starting NIU Image Stream Server in Docker container..."
echo "Press Ctrl+C to stop the server."

# Create data directory if it doesn't exist
mkdir -p data
mkdir -p logs

# Build the Docker image
echo "Building Docker image..."
docker build -t mjpeg-stream .

# Run the container
echo "Running Docker container..."
docker run -d \
  --name mjpeg-server \
  -p 8080:8080 \
  -v /home/sage/nfs/NIU:/home/sage/nfs/NIU:ro \
  -v "$(pwd)/data:/app/data" \
  mjpeg-stream

echo "Container started in detached mode. To view logs, run: docker logs -f mjpeg-server"
echo "To stop the container, run: ./stop-server.sh"
