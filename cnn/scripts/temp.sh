# Autojob v1.0
# Automatically retrieves code from source control and executes via SLURM
# Written by Xander Neuwirth, January 2020

# Create and descend hierarchy
mkdir -p autojobs
cd autojobs || exit

# Clear out previous duplicate jobs
rm -rf test_20210425_8:1307

# Pull down code
git clone git@github.com:AlexanderNeuwirth/FTIRCancerResearch.git test_20210425_8:1307
cd test_20210425_8:1307 || exit
git checkout test

# Schedule with SLURM
sbatch ./cnn/run.sh

# Give slurm time to create outfile
sleep 15

# Hook into updates
tail -f log.out
