#!/bin/bash
###########80#COLUMNS#BECAUSE#SOME#PEOPLE#STILL#USE#PUNCHCARDS#I#GUESS##########
#
# Example submit file for batch jobs on Rosie.
#
# To submit your job, run 'sbatch <jobfile>'
# To view your jobs in the Slurm queue, run 'squeue -l -u <your_username>'
# To view details of a running job, run 'scontrol show jobid -d <jobid>'
# To cancel a job, run 'scancel <jobid>'
#
# See the manpages for salloc, srun, sbatch, squeue, scontrol, and scancel
# for more information or read the Slurm docs online: https://slurm.schedmd.com
#
################################################################################
#
# command-line options to sbatch can be specified at the top of the batch
# submission file when preceeded by '#SBATCH'. These lines will be
# interpreted by the shell as comments but will be parsed by sbatch.
# These lines must be at the top of the file and may only be preceeded
# by comments and whitespace. See 'man sbatch' for a list of options.
#
# You _must_ specify the partition. Rosie's default is the 'teaching'
# partition for interactive nodes. You must use the 'batch' partition
# to submit jobs.
#SBATCH --partition=dgx
# The number of GPUs to request
#SBATCH --gpus=2
# The number of CPUs to request per GPU
#SBATCH --cpus-per-gpu=16
# Naming
#SBATCH --output=log.out
#SBATCH --job-name=test

# Activate the anaconda environment. Must use this form in scripts.
#. /usr/local/anaconda3/bin/activate
# Your job
python3 -m pip install --user virtualenv
python3 -m virtualenv venv

# Sourcing isn't working on the cluster with my configuration for some reason...
# source ./venv/bin/activate

./venv/bin/pip3 install -r requirements.txt

# Override PIL's maximum image size (must be done by modifying the library source)
/bin/sed -i -e 's/MAX_IMAGE_PIXELS =/MAX_IMAGE_PIXELS = 12 */' ./venv/lib/python3.6/site-packages/PIL/Image.py

./venv/bin/python3 -u main.py

# Deactivate the anaconda environment
# conda deactivate
