#!/bin/bash
#SBATCH --job-name=autoalbumentsearch
#SBATCH --output=/mnt/sandbox1/%u/logs/%A_%x
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=50:00:00
#SBATCH --mem-per-gpu=60G
source /home/alex.li/.bashrc
conda activate albumentations
export HYDRA_FULL_ERROR=1
autoalbument-search --config-dir /home/alex.li/git/scripts/autoalbument