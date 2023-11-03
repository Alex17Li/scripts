#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=/home/%u/logs/%A_%x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=150:00:00
source /home/$USER/.bashrc
conda activate cvml

cd /home/$USER/git

echo $COLUMNS
