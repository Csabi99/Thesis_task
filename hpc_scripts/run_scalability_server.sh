#!/bin/bash
#SBATCH --job-name=fedlearn-server
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=1-00:00:00
#SBATCH --partition=cpu
#SBATCH --output=/project/pr_fedlearn/logs/slurm-%j.out
#SBATCH --error=/project/pr_fedlearn/logs/slurm-%j.err
cd /project/pr_fedlearn/
module load singularity

LOGDIR="/project/pr_fedlearn/logs"
CONFIG="/project/pr_fedlearn/config/config.yaml"
IMAGE_CLIENT="/project/pr_fedlearn/hpc-client-image.sif"
IMAGE_SERVER="/project/pr_fedlearn/hpc-server-image.sif"

mkdir -p $LOGDIR

SERVER_NODE=$(hostname)
echo $SERVER_NODE > /project/pr_fedlearn/server_node.txt   # <-- WRITE HOSTNAME
echo "Starting server on $SERVER_NODE"
singularity exec --bind $LOGDIR:/app/logs --bind $CONFIG:/app/config.yaml $IMAGE_SERVER /opt/conda/bin/python /app/server.py > $LOGDIR/server.out 2>&1    
