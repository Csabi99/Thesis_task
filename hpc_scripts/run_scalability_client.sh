#!/bin/bash
#SBATCH --job-name=fedlearn-clients
#SBATCH --array=1-64
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --partition=cpu
#SBATCH --time=5:00:00
#SBATCH --exclude=x1000c1s5b0n1
#SBATCH --output=/project/pr_fedlearn/logs/slurm-%j.out
#SBATCH --error=/project/pr_fedlearn/logs/slurm-%j.err

cd /project/pr_fedlearn/
module load singularity
export SINGULARITY_IGNORE_USERNS=1
LOGDIR="/project/pr_fedlearn/logs"
CONFIG="/project/pr_fedlearn/config/config.yaml"
IMAGE_CLIENT="/project/pr_fedlearn/hpc-client-image.sif"
IMAGE_SERVER="/project/pr_fedlearn/hpc-server-image.sif"

mkdir -p $LOGDIR

# Block until file exists (server has started)
while [ ! -f /project/pr_fedlearn/server_node.txt ]; do
    sleep 1
done
delay=$(( SLURM_PROCID % 60 ))
sleep $delay
# Read server hostname
SERVER_NODE=$(cat /project/pr_fedlearn/server_node.txt)
SERVER_ADDRESS="${SERVER_NODE}:8080"

CLIENT_ID=$SLURM_ARRAY_TASK_ID
max_tries=5
for i in $(seq 1 $max_tries); do
    singularity exec   --bind /project/pr_fedlearn/passwd:/etc/passwd:ro --bind /project/pr_fedlearn/group:/etc/group:ro --bind $LOGDIR:/app/logs --bind $CONFIG:/app/config.yaml $IMAGE_CLIENT /opt/conda/bin/python /app/client.py --num=$CLIENT_ID --server_address=$SERVER_ADDRESS > $LOGDIR/client${CLIENT_ID}.out 2>&1 && break
    echo "Retry $i on host $(hostname)" >> retry.log
    sleep 5
done

