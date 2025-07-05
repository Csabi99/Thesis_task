#!/bin/bash

# Check if both arguments are provided
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <path> <namespace>"
  exit 1
fi

# Define variables from input arguments
path="$1"
ns="$2"
config_file="$path/config.yaml"

echo "Starting script execution with path: $path and namespace: $ns..."
echo "Navigating to the working directory at /home/ubuntu/data/..."
cd /home/ubuntu/data/ || exit

# Copy the configuration file
echo "Copying configuration file from ${path}/config.yaml to /home/ubuntu/data/config.yaml..."
cp "${path}/config.yaml" /home/ubuntu/data/config.yaml

# Build and push Docker images
echo "Building and pushing Docker images..."
docker build -t ccsaba99/server-image:latest -f Dockerfile_server .
docker build -t ccsaba99/client-image:latest -f Dockerfile_client .
docker push ccsaba99/server-image:latest
docker push ccsaba99/client-image:latest

# Kubernetes setup
echo "Setting up Kubernetes namespace $ns and applying persistent volumes..."
sudo kubectl create namespace "$ns"
sudo kubectl apply -f persistent-volume.yaml
sudo kubectl apply -f persistent-volume-claim.yaml -n "$ns"

# Apply server configurations
echo "Applying server configurations..."
cd "${path}/job_configs" || exit
sudo kubectl apply -f server-service.yaml -n "$ns"
sudo kubectl apply -f server_job.yaml -n "$ns"

# echo "Waiting for server setup to stabilize (45 seconds)..."
# sleep 45

# # Loop to apply client jobs
# echo "Applying client job configurations for $clients clients..."
# for i in $(seq 1 "$clients"); do
#     echo "Applying client-job-$i.yaml..."
#     sudo kubectl apply -f "client-job-$i.yaml" -n "$ns"
# done

echo "Script execution completed."
