#!/bin/bash

#SBATCH -n 2
#SBATCH --mem=16G
#SBATCH -t 48:00:00
#SBATCH --partition=3090-gcondo --gres=gpu:1
#SBATCH -o run_ddpm.out

module load python/3.8.12_gcc8.3
module load gcc/8.3

module load cuda/11.1.1
module load cudnn/8.2.0

#cd ~/data/yliang51/envs

#rm -rf MOSO
#virtualenv -p python3 MOSO

#source ~/data/yliang51/envs/MOSO/bin/activate

module load anaconda/3-5.2.0
source /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh

#conda remove --prefix ~/data/yliang51/envs/pvdm --all
#conda create --prefix ~/data/yliang51/envs/pvdm python=3.7 -y
conda activate ~/data/yliang51/envs/pvdm

cd ..

python main.py \
 --exp ddpm \
 --id main \
 --pretrain_config configs/latent-diffusion/base.yaml \
 --data UCF101 \
 --first_model 'results/first_stage_main_gan_UCF101_42/model_last.pth'  
 --diffusion_config configs/latent-diffusion/base.yaml \
 --batch_size 7





