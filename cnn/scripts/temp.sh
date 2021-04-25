# Autojob v1.0
# Automatically retrieves code from source control and executes via SLURM
# Written by Xander Neuwirth, January 2020

# Create and descend hierarchy
mkdir -p autojobs
cd autojobs || exit

# Clear out previous duplicate jobs
rm -rf test_20210425_8:3120

# Pull down code
git clone git@github.com:AlexanderNeuwirth/FTIRCancerResearch.git test_20210425_8:3120
cd test_20210425_8:3120 || exit
git checkout test

# Schedule with SLURM
srun singularity shell --nv -B /data/:/data/ /data/containers/msoe-tensorflow.sif ./cnn/run.sh

# Give slurm time to create outfile
sleep 15

# Hook into updates
tail -f log.out
