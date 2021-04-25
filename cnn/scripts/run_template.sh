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
#SBATCH --job-name={{BRANCH}}

# Set up venv
(
cd cnn || exit
python3 -m pip install --user virtualenv
python3 -m virtualenv venv
./venv/bin/pip3 install -r requirements.txt
)

# Run scripts
data="/data/berisha_lab/neuwirth/data/mnf16"
masks="/data/berisha_lab/neuwirth/annotations_4"
checkpoint="adam0p01"
crop=33
classes=5
samples=100000
epochs=1
batch=128

./cnn/venv/bin/python3 -u cnn/hsi_cnn/hsi_cnn_train_keras.py --data $data --masks $masks/train --crop $crop --checkpoint $checkpoint --classes $classes --epochs $epochs --batch $batch --balance --samples $samples --validate --valdata $data --valmasks $masks/val --valsamples $samples

./cnn/venv/bin/python3 -u cnn/hsi_cnn/hsi_cnn_classify_keras.py --data $data --masks $masks/test --checkpoint $checkpoint --crop $crop --classes $classes
