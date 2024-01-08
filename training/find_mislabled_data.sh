#!/bin/bash
#SBATCH --job-name=find_mislabled
#SBATCH --output=/home/%u/logs/%A_%x
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --time=150:00:00

source /home/$USER/.bashrc
conda activate cvml

cd /home/$USER/git/JupiterCVML

export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_MIN_CHANNELS=32
export COLUMNS=100

srun --kill-on-bad-exit python -m kore.scripts.seg_find_mislabeled_data \
    --warm_up_steps 1000 \
    --trainer.callbacks.tqdm false \
    --trainer.logger.version find_mislabeled_data
