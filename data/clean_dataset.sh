#!/bin/bash
#SBATCH --job-name=clean_dataset
#SBATCH --output=/mnt/sandbox1/%u/logs/%A_%x.txt
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@bluerivertech.com

source ~/.bashrc

cd /home/${USER}/git/JupiterCVML

python clean_dataset.py europa/base/src/europa/dl/config/halo_masks/create_halo_masks.py
