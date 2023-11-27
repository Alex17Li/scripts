#!/bin/bash
#SBATCH --job-name=jup_alex
#SBATCH --output=/home/%u/logs/%A_%x
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=8:00:00

source /home/$USER/.bashrc
conda activate cvml

cd /home/$USER/git

jupyter notebook --no-browser --port=8989 --ip=0.0.0.0
