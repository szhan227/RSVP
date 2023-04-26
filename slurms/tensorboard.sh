#!/bin/bash

#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -t 1:00:00
#SBATCH --job-name tensorboard
#SBATCH --output tensorboard.out


ipnport=$(shuf -i8000-9999 -n1)
ipnip=$(hostname -i)

module load python/3.7.4

module load gcc/10.2
module load cuda/11.3.1
module load cudnn/8.2.0

source ~/data/yliang51/envs/nrnerf/bin/activate


tensorboard --logdir="../results/ddpm_main_UCF101_42" --port=$ipnport --host=$ipnip
