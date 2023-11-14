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
# EXP=16686
CKPT_PATH=/mnt/sandbox1/alex.li/wandb/run-20231110_200223-16686/files/epoch\=35-val_loss\=0.113822.ckpt
# CKPT_PATH=/mnt/sandbox1/alex.li/wandb/run-20231113_133040-16754/files/epoch=5-val_loss=0.128479.ckpt
# CKPT_PATH=null
EXP=16686

set -x

CONFIG_PATH="scripts/kore_configs/harvest_seg_train.yml"
srun --kill-on-bad-exit python -m JupiterCVML.kore.scripts.train_seg \
    --ckpt_path $CKPT_PATH \
    --config_path $CONFIG_PATH \
    --lr_scheduler.T_max 100 \
    --trainer.logger.version $EXP


# CONFIG_PATH="scripts/kore_configs/harvest_seg_train.yml scripts/kore_configs/seg_gsam.yml"
# srun --kill-on-bad-exit python -m JupiterCVML.kore.scripts.train_seg \
#     --config_path $CONFIG_PATH \
#     --data.train_set.csv_name master_annotations_10k.csv \
#     --trainer.logger.version $EXP
