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

cd /home/$USER/git/JupiterCVML

export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_MIN_CHANNELS=32
export COLUMNS=100

EXP=${SLURM_JOB_ID}
# CKPT_PATH=/mnt/sandbox1/alex.li/wandb/run-18495/files/epoch=9-val_loss=0.069723.ckpt
CKPT_PATH=/mnt/sandbox1/alex.li/models/18495.ckpt

# CONFIG_PATH="scripts/kore_configs/harvest_seg_train.yml scripts/kore_configs/seg_gsam.yml \$CVML_DIR/koreconfigs/options/seg_no_dust_head.yml"

set -x

srun --kill-on-bad-exit python -m kore.scripts.train_seg \
    --config_path \$CVML_DIR/kore/configs/options/seg_no_dust_head.yml \
    --ckpt_path $CKPT_PATH \
    --optimizer GSAM \
    --optimizer.rho_min 0.0001 \
    --optimizer.rho_max 0.001 \
    --optimizer.alpha 0.3 \
    --optimizer.adaptive False \
    --optimizer.lr 4e-4 \
    --augmentation.albumentation_transform_path \$CVML_DIR/kore/configs/data/albumentations/seg_trivialaugment.yml \
    --finetuning.skip_mismatched_layers True \
    --trainer.callbacks.tqdm False \
    --trainer.logger.version $EXP \
    --trainer.callbacks.early_stopping.patience 100
