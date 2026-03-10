#!/bin/bash
#SBATCH --job-name=fedlearn-server
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/project/nr_fedlearn/logs/slurm-%j.out
#SBATCH --error=/project/nr_fedlearn/logs/slurm-%j.err
# export WANDB_API_KEY=
cd /project/nr_fedlearn/
module load singularity
LOGDIR="/project/nr_fedlearn/logs"
CONFIG="/project/nr_fedlearn/config/config.yaml"
IMAGE_CLIENT="/project/nr_fedlearn/hpc-client-image.sif"
IMAGE_SERVER="/project/nr_fedlearn/hpc-server-image.sif"

mkdir -p $LOGDIR

singularity exec --nv --bind $LOGDIR:/app/logs --bind $CONFIG:/app/config.yaml --bind /project/nr_fedlearn/strategies.py:/app/strategies.py --bind /project/nr_fedlearn/base_training.py:/app/base_training.py $IMAGE_SERVER /opt/conda/bin/python /app/base_training.py > $LOGDIR/base.out 2>&1    
