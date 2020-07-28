#!/bin/bash

# The SBATCH directives must appear before any executable line in this script.

#SBATCH --qos medium       # QOS (priority).
#SBATCH -N 1               # Number of nodes requested.
#SBATCH -n 8               # Number of tasks (i.e. processes).
#SBATCH --cpus-per-task=1  # Number of cores per task.
#SBATCH --gres=gpu:1       # Number of GPUs.
##SBATCH --nodelist=em1    # Uncomment if you need a specific machine.

# Uncomment this to have Slurm cd to a directory before running the script.
# You can also just run the script from the directory you want to be in.
#SBATCH -D /home/nio/HyperRIM/codes/

# Uncomment to control the output files. By default stdout and stderr go to
# the same place, but if you use both commands below they'll be split up.
# %N is the hostname (if used, will create output(s) per node).
# %j is jobid.
#SBATCH -o ./logs/slurm.train_hrim_but_x4_0.out    # STDOUT
##SBATCH -e slurm.%N.%j.err    # STDERR

# Print some info for context.
pwd
hostname
date

echo "Starting job..."

source ~/.bashrc
conda activate py37

# Python will buffer output of your script unless you set this.
# If you're not using python, figure out how to turn off output
# buffering when stdout is a file, or else when watching your output
# script you'll only get updated every several lines printed.
export PYTHONUNBUFFERED=1

# Do all the research.
python train_ms_two.py -opt options/train/train_butterfly_32to128.json

# Print completion time.
date