# Autojob v1.0
# Automatically retrieves code from source control and executes via SLURM
# Written by Xander Neuwirth, January 2020

# Create and descend hierarchy
mkdir -p autojobs
cd autojobs || exit

# Clear out previous duplicate jobs
rm -rf {{ID}}

# Pull down code
git clone {{REMOTE}} {{ID}}
cd {{ID}} || exit
git checkout {{BRANCH}}

# Schedule with SLURM
srun -w dh-node19 --pty --gpus=2 --cpus-per-gpu=8 singularity exec --nv -B /data/:/data/ /data/containers/msoe-tensorflow.sif ./cnn/run.sh

# Give slurm time to create outfile
sleep 15

# Hook into updates
tail -f log.out