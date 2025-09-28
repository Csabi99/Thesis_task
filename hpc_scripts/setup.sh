docker build -t ccsaba99/hpc-server-image:latest -f Dockerfile_server .
docker build -t ccsaba99/hpc-client-image:latest -f Dockerfile_client .
docker push ccsaba99/hpc-server-image:latest
docker push ccsaba99/hpc-client-image:latest


srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem=64G --time=02:00:00 --partition=cpu --account=pr_fedlearn --pty bash


srun --nodes=2 --ntasks=2 --cpus-per-task=64 --mem=64G --time=02:00:00 --partition=cpu --account=pr_fedlearn --pty bash
module load singularity
scontrol show hostnames $SLURM_NODELIST
export SINGULARITY_TMPDIR=/project/pr_fedlearn/singularity_tmp
export SINGULARITY_CACHEDIR=/project/pr_fedlearn/cache
mkdir -p $SINGULARITY_TMPDIR $SINGULARITY_CACHEDIR
DOCKER_IMAGE_SERVER="ccsaba99/hpc-server-image"
IMAGE_SERVER="/project/pr_fedlearn//hpc-server-image.sif"
singularity pull "$IMAGE_SERVER" "docker://$DOCKER_IMAGE_SERVER"
DOCKER_IMAGE_CLIENT="ccsaba99/hpc-client-image"
IMAGE_CLIENT="/project/pr_fedlearn//hpc-client-image.sif"
singularity pull "$IMAGE_CLIENT" "docker://$DOCKER_IMAGE_CLIENT"
LOGDIR="/project/pr_fedlearn/logs"
mkdir -p $LOGDIR

#use ssh to host instead, set the variables and run the commands
NODE0=x1000c1s2b1n0
ssh $NODE0 "singularity exec --bind $LOGDIR:/app/logs $IMAGE_SERVER nohup python /app/server.py > $LOGDIR/server.out 2>&1 & echo \$! > $LOGDIR/server.pid"
SERVER_ADDRESS="$NODE0:8080"
NODE1=x1000c1s2b1n1
ssh $NODE1 "singularity exec --bind $LOGDIR:/app/logs $IMAGE_CLIENT nohup python /app/client.py --num=1 --server_address=$SERVER_ADDRESS > $LOGDIR/client1.out 2>&1 & echo \$! > $LOGDIR/client1.pid"
ssh $NODE1 "singularity exec --bind $LOGDIR:/app/logs $IMAGE_CLIENT nohup python /app/client.py --num=2 --server_address=$SERVER_ADDRESS > $LOGDIR/client2.out 2>&1 & echo \$! > $LOGDIR/client2.pid"

singularity exec --bind $LOGDIR:/app/logs $IMAGE_SERVER nohup python /app/server.py > $LOGDIR/server.out 2>&1 & echo $! > $LOGDIR/server.pid
singularity exec --bind $LOGDIR:/app/logs $IMAGE_CLIENT nohup python /app/client.py --num=1 > $LOGDIR/client1.out 2>&1 & echo $! > $LOGDIR/client1.pid
singularity exec --bind $LOGDIR:/app/logs $IMAGE_CLIENT nohup python /app/client.py --num=2 > $LOGDIR/client2.out 2>&1 & echo $! > $LOGDIR/client2.pid

#https://chatgpt.com/c/68d2e8aa-b978-8333-85ca-839ffc7520b0