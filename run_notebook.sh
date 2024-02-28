#!/bin/bash
#SBATCH --job-name=jup_alex
#SBATCH --output=/mnt/sandbox1/%u/logs/%A_%x
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=8:00:00
#SBATCH --mem-per-gpu=10G

# SBATCH --partition=cpu
# SBATCH --ntasks=1
# SBATCH --cpus-per-task=16

source /home/$USER/.bashrc
conda activate cvml

cd /home/$USER/git

jupyter notebook --no-browser --port=8989 --ip=0.0.0.0
