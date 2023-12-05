#!/bin/bash
#SBATCH --job-name=seg_dust
#SBATCH --output=/home/%u/logs/%A_%x
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --time=150:00:00

source /home/$USER/.bashrc
conda activate cvml

cd /home/$USER/git

export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_MIN_CHANNELS=32
export COLUMNS=100

EXP=${SLURM_JOB_ID}
# CKPT_PATH=/mnt/sandbox1/alex.li/wandb/run-16903/files/epoch=95-val_loss=0.084565.ckpt
CKPT_PATH=/mnt/sandbox1/alex.li/wandb/run-17866/files/last.ckpt
set -x

srun --kill-on-bad-exit python -m JupiterCVML.kore.scripts.train_seg \
    --ckpt_path $CKPT_PATH \
    --finetuning.skip_mismatched_layers true \
    --trainer.logger.version $EXP

CONFIG_PATH="scripts/kore_configs/harvest_seg_train.yml"

# CONFIG_PATH="scripts/kore_configs/harvest_seg_train.yml scripts/kore_configs/seg_gsam.yml"
# srun --kill-on-bad-exit python -m JupiterCVML.kore.scripts.train_seg \
#     --config_path $CONFIG_PATH \
#     --ckpt_path $CKPT_PATH \
#     --trainer.logger.version $EXP
