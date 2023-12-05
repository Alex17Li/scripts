#!/bin/bash
#SBATCH --job-name=seg_dust
#SBATCH --output=/home/%u/logs/%A_%x
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=150:00:00

source /home/$USER/.bashrc
conda activate cvml

cd /home/$USER/git

export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_MIN_CHANNELS=32
export COLUMNS=100

EXP=${SLURM_JOB_ID}_1gpu
CKPT_PATH=/mnt/sandbox1/alex.li/wandb/run-17866/files/last.ckpt
set -x

srun --kill-on-bad-exit python -m JupiterCVML.kore.scripts.train_seg \
    --ckpt_path $CKPT_PATH \
    --batch_size 64 \
    --finetuning.skip_mismatched_layers true \
    --trainer.logger.version $EXP
