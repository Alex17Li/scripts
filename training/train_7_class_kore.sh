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
# --
    # --augmentation.albumentation_transform_path \$CVML_DIR/kore/configs/data/albumentations/seg_trivialaugment.yml \
#     --lr_scheduler EXP \
#     --lr_scheduler.end_factor 1e-3 \

srun --kill-on-bad-exit python -m kore.scripts.train_seg \
    --config_path /home/alexli/git/scripts/kore_configs/gpu_seg.yml \
    --ckpt_path $CKPT_PATH \
    --run-id $EXP \
    --optimizer AdamW \
    --optimizer.lr 5e-2 \
    --trainer.strategy.find_unused_parameters true \
    --finetuning.skip_mismatched_layers true \
    --trainer.callbacks.tqdm false \
    --trainer.enable_early_stopping false \
    --model.model_params.structural_reparameterization_on_stem true \
    --augmentation.cnp.humans.blend-mode VANILLA \
    --augmentation.cnp.humans.depth_aware true \
    --augmentation.cnp.humans.only_non_occluded false \
    --augmentation.cnp.humans.jitter_object false \
    --augmentation.cnp.humans.jitter_range 0.2 \
    --augmentation.cnp.humans.sample_ratio 0.3
