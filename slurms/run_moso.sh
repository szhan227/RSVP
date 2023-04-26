#!/bin/bash

#SBATCH -n 2
#SBATCH --mem=16G
#SBATCH -t 48:00:00
#SBATCH --partition=3090-gcondo --gres=gpu:1
#SBATCH -o run_moso.out

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


#conda install -c menpo opencv

#pip install pytorch-msssim

cd ..
rm -rf ./results/moso_ddpm_main_UCF101_42/

python main.py \
 --exp moso_ddpm \
 --id main \
 --data UCF101 \
 --diffusion_config configs/moso-diffusion/base.yaml \
 --moso_config '../MOSO/MOSO-VQVAE/config/test_UCF.yaml' \
 --moso_checkpoint '../MOSO/MOSO-VQVAE/experiments/MoCoVQVAEwCDsCB_UCF_im256_16frames_id4_2023-04-10-15-48-14/MoCoVQVAE_wCD_shareCB_iter65000.pth'



