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
CKPT_PATH=/mnt/sandbox1/alex.li/wandb/run-18938/files/epoch=9-val_loss=0.066906.ckpt

# CONFIG_PATH="scripts/kore_configs/harvest_seg_train.yml scripts/kore_configs/seg_gsam.yml \$CVML_DIR/koreconfigs/options/seg_no_dust_head.yml"

set -x

# --optimizer GSAM \
# --optimizer.rho_min .002 \
# --optimizer.rho_max .02 \
# --optimizer.alpha .3 \

# --optimizer.SAM \
# --optimizer.rho .02 \

# --optimizer AdamW \

srun --kill-on-bad-exit python -m kore.scripts.train_seg \
    --ckpt_path $CKPT_PATH \
    --optimizer AdamW \
    --optimizer.lr 4e-4 \
    --trainer.strategy.find_unused_parameters true \
    --finetuning.skip_mismatched_layers true \
    --trainer.callbacks.tqdm false \
    --trainer.logger.version $EXP \
    --trainer.callbacks.early_stopping.patience 100 \
    --augmentation.cnp.humans.depth_aware true \
    --augmentation.cnp.humans.only_non_occluded false \
    --augmentation.cnp.humans.jitter_object true \
    --augmentation.cnp.humans.jitter_range 0.2 \
    --augmentation.cnp.humans.sample_ratio 0.3
