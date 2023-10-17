#!/bin/bash
#SBATCH --job-name=seg_dust
#SBATCH --output=/home/%u/logs/%A_%x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --time=150:00:00

#--SBATCH --partition=cpu
source /home/$USER/.bashrc
conda activate cvml

cd /home/$USER/git
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_MIN_CHANNELS=32

export COLUMNS=200

python -m JupiterCVML.europa.base.src.europa.dlcore.scripts.train_seg \
    --trainer.precision '16-mixed'
