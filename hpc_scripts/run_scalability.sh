#!/bin/bash
#SBATCH --job-name=federated-learning-scalability-test
#SBATCH --nodes=3                  # number of nodes to use
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --time=05:00:00
#SBATCH --partition=cpu
#SBATCH --account=pr_fedlearn
#SBATCH --output=/project/pr_fedlearn/logs/slurm-%j.out
#SBATCH --error=/project/pr_fedlearn/logs/slurm-%j.err

# ------------------------------
# Configurable parameters
# ------------------------------
TOTAL_CLIENTS=${TOTAL_CLIENTS:-16}   # default 16 if not provided
CONFIG=${CONFIG:-/project/pr_fedlearn/config/config.yaml}

# ------------------------------
# Setup
# ------------------------------
cd /project/pr_fedlearn/
module load singularity

export SINGULARITY_TMPDIR=/project/pr_fedlearn/singularity_tmp
export SINGULARITY_CACHEDIR=/project/pr_fedlearn/cache
mkdir -p $SINGULARITY_TMPDIR $SINGULARITY_CACHEDIR

DOCKER_IMAGE_SERVER="ccsaba99/hpc-server-image"
IMAGE_SERVER="/project/pr_fedlearn/hpc-server-image.sif"
[ ! -f "$IMAGE_SERVER" ] && singularity pull "$IMAGE_SERVER" "docker://$DOCKER_IMAGE_SERVER"

DOCKER_IMAGE_CLIENT="ccsaba99/hpc-client-image"
IMAGE_CLIENT="/project/pr_fedlearn/hpc-client-image.sif"
[ ! -f "$IMAGE_CLIENT" ] && singularity pull "$IMAGE_CLIENT" "docker://$DOCKER_IMAGE_CLIENT"

LOGDIR="/project/pr_fedlearn/logs"
mkdir -p $LOGDIR

# ------------------------------
# Launch server on first node
# ------------------------------
NODELIST=($(scontrol show hostnames $SLURM_NODELIST))
SERVER_NODE=${NODELIST[0]}
NODECOUNT=${#NODELIST[@]}

echo "Starting server on $SERVER_NODE"
ssh $SERVER_NODE "module load singularity && nohup singularity exec --bind $LOGDIR:/app/logs --bind $CONFIG:/app/config.yaml $IMAGE_SERVER python /app/server.py > $LOGDIR/server.out 2>&1 & disown; echo \$! > $LOGDIR/server.pid" < /dev/null > /dev/null 2>&1 &

# Give server time to start
sleep 15

# ------------------------------
# Launch clients across nodes
# ------------------------------
echo "Starting $TOTAL_CLIENTS clients across $NODECOUNT nodes"

i=1
SERVER_ADDRESS="$SERVER_NODE:8080"
for NODE in "${NODELIST[@]}"; do
  # how many clients should run on this node?
  CLIENTS_PER_NODE=$(( (TOTAL_CLIENTS + NODECOUNT - 1) / NODECOUNT ))  # ceil division
  for j in $(seq 1 $CLIENTS_PER_NODE); do
    if [ $i -gt $TOTAL_CLIENTS ]; then
      break
    fi
    ssh $NODE "module load singularity && nohup singularity exec --bind $LOGDIR:/app/logs --bind $CONFIG:/app/config.yaml $IMAGE_CLIENT python /app/client.py --num=$i --server_address=$SERVER_ADDRESS > $LOGDIR/client${i}.out 2>&1 & disown" < /dev/null &
    echo "Client $i started on node $NODE"
    i=$((i+1))
  done
done

wait
echo "All processes launched."

# run on multiple nodes
#if you want to run interactively, use:
#srun --nodes=3 --ntasks-per-node=1 --cpus-per-task=128 --mem=200G --time=05:00:00 --partition=cpu --account=pr_fedlearn --pty bash