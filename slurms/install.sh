#!/bin/bash

#SBATCH -n 2
#SBATCH --mem=16G
#SBATCH -t 48:00:00
#SBATCH --partition=a6000-gcondo --gres=gpu:1
#SBATCH -o install.out

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

conda remove --prefix ~/data/yliang51/envs/pvdm --all
conda create --prefix ~/data/yliang51/envs/pvdm python=3.7 -y
conda activate ~/data/yliang51/envs/pvdm
conda install pip
which pip

pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

python -c "import torch; print(torch.cuda.is_available())"

pip install natsort tqdm gdown omegaconf einops lpips pyspng tensorboard imageio av moviepy

conda install -c menpo opencv
