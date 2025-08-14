#!/bin/bash

# Stop the running Docker container
echo "Stopping MJPEG Stream Server container..."
docker stop mjpeg-server

# Remove the container after stopping
echo "Removing container..."
docker rm mjpeg-server

echo "Server stopped and container removed."
