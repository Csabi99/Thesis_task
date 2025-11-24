#!/bin/bash
#SBATCH --job-name=fl_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=228G

#SBATCH --time=10:00:00
#SBATCH --partition=cpu
#SBATCH --account=pr_fedlearn
#SBATCH --output=/project/pr_fedlearn/logs/slurm-%j.out
#SBATCH --error=/project/pr_fedlearn/logs/slurm-%j.err

#--cpus-per-task=64 --gres=gpu:1

# ------------------------------
# Setup
# ------------------------------
module load singularity
LOGDIR="/project/pr_fedlearn/logs"
CONFIG="/project/pr_fedlearn/config/config.yaml"
IMAGE_CLIENT="/project/pr_fedlearn/hpc-client-image.sif"
IMAGE_SERVER="/project/pr_fedlearn/hpc-server-image.sif"

mkdir -p $LOGDIR

# ------------------------------
# Start server
# ------------------------------
echo "Starting server..."
singularity exec --bind $LOGDIR:/app/logs --bind $CONFIG:/app/config.yaml $IMAGE_SERVER nohup /opt/conda/bin/python /app/server.py > $LOGDIR/server.out 2>&1 & echo $! > $LOGDIR/server.pid
# --nv
# Give server time to start
sleep 15
HOSTIP=$(hostname -I | awk '{print $1}')


# ------------------------------
# Start 64 clients
# ------------------------------
for i in $(seq 1 64); do
  echo "Starting client $i..."
  singularity exec --bind $LOGDIR:/app/logs --bind $CONFIG:/app/config.yaml $IMAGE_CLIENT nohup /opt/conda/bin/python /app/client.py --num=$i --server_address $HOSTIP:8080 > $LOGDIR/client${i}.out 2>&1 & echo $! > $LOGDIR/client${i}.pid
done
# --nv
wait
echo "Server and clients ran."

#start with sbatch run_fl.sh
# run on one node with gpu