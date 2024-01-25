#!/bin/bash
#SBATCH --job-name=r1_seg
#SBATCH --output=/home/%u/logs/%A_%x
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --time=200:00:00
#SBATCH --mem-per-gpu=60G

source /home/$USER/.bashrc
conda activate cvml

cd /home/$USER/git/JupiterCVML

export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_MIN_CHANNELS=32
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CILK_NWORKERS=1
export TBB_MAX_NUM_THREADS=1
export PL_GLOBAL_SEED=304
export COLUMNS=100

EXP=${SLURM_JOB_ID}

# CONFIG_PATH="scripts/kore_configs/harvest_seg_train.yml scripts/kore_configs/seg_gsam.yml \$CVML_DIR/koreconfigs/options/seg_no_dust_head.yml"

set -x

# --optimizer GSAM \
# --optimizer.rho_min .002 \
# --optimizer.rho_max .02 \
# --optimizer.alpha .3 \

# --optimizer.SAM \
# --optimizer.rho .02 \

# --optimizer AdamW \
# --augmentation.albumentation_transform_path \$CVML_DIR/kore/configs/data/albumentations/seg_trivialaugment.yml \
#     --lr_scheduler EXP \
#     --lr_scheduler.end_factor 1e-3 \
#     --ckpt_path $CKPT_PATH \

# srun --kill-on-bad-exit python -m kore.scripts.train_seg \
#     --run-id $EXP \
#     --optimizer AdamW \
#     --optimizer.lr 1e-3 \
#     --trainer.callbacks.tqdm false \
#     --trainer.precision 32 \
#     --trainer.enable_early_stopping false

# Test on subset
# srun --kill-on-bad-exit python -m kore.scripts.train_seg \
#     --config_path "\$CVML_DIR/kore/configs/options/seg_no_dust_head.yml" \
#     --run-id adam_$EXP \
#     --warm_up_steps 1000 \
#     --optimizer AdamW \
#     --lr_scheduler EXP \
#     --lr_scheduler.end_factor 5e-3 \
#     --trainer.max_epochs 25 \
#     --data.train_set.csv master_annotations_40k.csv \
#     --data.validation_set_ratio 0.05 \
#     --augmentation.albumentation_transform_path \$CVML_DIR/kore/configs/data/albumentations/seg_trivialaugment.yml \
#     --optimizer.lr 1e-2 \
#     --trainer.callbacks.tqdm false \
#     --trainer.precision 32 \
#     --trainer.enable_early_stopping false

# Current strategy
# srun --kill-on-bad-exit python -m kore.scripts.train_seg \
#     --run-id $EXP \
#     --lr_scheduler EXP \
#     --lr_scheduler.end_factor 1e-3 \
#     --augmentation.albumentation_transform_path \$CVML_DIR/kore/configs/data/albumentations/seg_trivialaugment.yml \
#     --optimizer AdamW \
#     --optimizer.lr 5e-2 \
#     --optimizer.weight_decay 7e-4 \
#     --trainer.callbacks.tqdm false \
#     --trainer.precision 32 \
#     --trainer.enable_early_stopping false \
#     --trainer.max_epochs 30 \
#     --batch_size 12 \
#     --warm_up_steps 1000 \
#     --model.model_params.structural_reparameterization_on_stem true \
#     --output_dir /mnt/sandbox1/$USER/train_rev1/\$RUN_ID

# INFINI RUN
srun --kill-on-bad-exit python -m kore.scripts.train_seg \
    --run-id ${EXP}_rev1_overfit \
    --lr_scheduler EXP \
    --lr_scheduler.end_factor 1e-2 \
    --augmentation.albumentation_transform_path \$CVML_DIR/kore/configs/data/albumentations/seg_trivialaugment.yml \
    --optimizer AdamW \
    --optimizer.lr 6e-3 \
    --optimizer.weight_decay 6e-3 \
    --trainer.callbacks.tqdm false \
    --trainer.precision 32 \
    --trainer.enable_early_stopping false \
    --trainer.max_epochs 50 \
    --trainer.accumulate_grad_batches 3 \
    --batch_size 12 \
    --warm_up_steps 1000 \
    --model.model_params.structural_reparameterization_on_stem true \
    --ckpt_path /mnt/sandbox1/alex.li/train_rev1/20446/checkpoints/epoch=49-val_loss=0.091303.ckpt \
    --finetuning.enable true \
    --output_dir /mnt/sandbox1/$USER/train_rev1/\$RUN_ID \
    --data.validation_set.csv v6_2_overlap_with_test_geohash_bag_vat_ids.csv \
    --data.validation_set.dataset_path /data2/jupiter/datasets/Jupiter_train_v6_2 \
    --data.validation_set.absolute_csv false
