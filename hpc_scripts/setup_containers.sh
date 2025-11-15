docker build -t ccsaba99/hpc-server-image:latest -f Dockerfile_server .
docker build -t ccsaba99/hpc-client-image:latest -f Dockerfile_client .
docker push ccsaba99/hpc-server-image:latest
docker push ccsaba99/hpc-client-image:latest

# next step is to run the scalability tests with sbatch run_scalability.sh on gpu
