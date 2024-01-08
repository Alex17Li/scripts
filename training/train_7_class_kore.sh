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

EXP=$SLURM_JOB_ID
# CKPT_PATH=/mnt/sandbox1/alex.li/wandb/run-18495/files/epoch=9-val_loss=0.069723.ckpt
# CONFIG_PATH="scripts/kore_configs/harvest_seg_train.yml scripts/kore_configs/seg_gsam.yml \$CVML_DIR/koreconfigs/options/seg_no_dust_head.yml"

set -x

# --optimizer GSAM \
# --optimizer.rho_min .002 \
# --optimizer.rho_max .02 \
# --optimizer.alpha .3 \

# --optimizer SAM \
# --optimizer.rho .02 \
# --trainer.strategy.ddp_find_unused_parameters True \

# --optimizer AdamW \

# --lr_scheduler NONE
# --augmentation.albumentation_transform_path \$CVML_DIR/kore/configs/data/albumentations/seg_trivialaugment.yml \
#  --warm_up_steps 0 \
srun --kill-on-bad-exit python -m kore.scripts.seg_find_mislabled_data \
    --warm_up_steps 20 \
    --trainer.callbacks.tqdm false \
    --trainer.logger.version $EXP


srun --kill-on-bad-exit python -m kore.scripts.train_seg \
    --warm_up_steps 20 \
    --trainer.callbacks.tqdm false \
    --trainer.logger.version $EXP \


# srun --kill-on-bad-exit python -m kore.scripts.train_seg \
#     --optimizer AdamW \
#     --optimizer.lr 5e-2 \
#     --lr_scheduler EXP \
#     --lr_scheduler.end_factor 1e-3 \
#     --trainer.callbacks.tqdm false \
#     --trainer.logger.version $EXP \
#     --trainer.enable_early_stopping false \
#     --model.model_params.structural_reparameterization_on_stem true \
#     --augmentation.albumentation_transform_path \$CVML_DIR/kore/configs/data/albumentations/seg_trivialaugment.yml \
#     --trainer.max_epochs 300 \
#     --batch_size 16 \
#     --ckpt_path /mnt/sandbox1/alex.li/run-19271/files/last.ckpt \
#     --warm_up_steps 10000
