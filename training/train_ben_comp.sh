#!/bin/bash
#SBATCH --job-name=r2_train_seg
#SBATCH --output=/home/%u/logs/%A_%x
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --time=150:00:00
#SBATCH --exclude=stc01sppamxnl004
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

set -x
    # --augmentation.albumentation_transform_path \$CVML_DIR/kore/configs/data/albumentations/seg_trivialaugment_halo.yml \
    # --augmentation.albumentation_transform_path \$CVML_DIR/kore/configs/data/albumentations/seg_halo.yml \

# /home/alex.li/git/scripts/training/dustaug.yml
# srun --kill-on-bad-exit python -m kore.scripts.train_seg \
#     --optimizer.weight_decay 1e-3 \
#     --config_path /home/alex.li/git/scripts/training/halo_7_class_train.yml /home/alex.li/git/scripts/training/dustaug.yml \
#     --augmentation.albumentation_transform_path \$CVML_DIR/kore/configs/data/albumentations/seg_halo.yml \
#     --augmentation.use_cutmix true \
#     --trainer.precision 16-mixed \
#     --optimizer.lr 1e-3 \
#     --run_id ${EXP}_r2_rgb_40k_dust \
#     --output_dir /mnt/sandbox1/$USER/train_halo/$RUN_ID \
#     --trainer.max_epochs 100

#  /home/alex.li/git/scripts/training/dustaug.yml \
# srun --kill-on-bad-exit python -m kore.scripts.train_seg \
#     --trainer.precision 16-mixed \
#     --optimizer.weight_decay 1e-3 \
#     --config_path /home/alex.li/git/scripts/training/halo_7_class_train.yml\
#     --trainer.enable_early_stopping false \
#     --output_dir /mnt/sandbox1/$USER/train_rev1/\$RUN_ID \
#     --run-id ${EXP}_re2_downsampling_40k
    # --model.model_params.use_highres_downsampling true \
srun --kill-on-bad-exit python -m kore.scripts.train_seg \
    --config_path kore/configs/defaults/halo_seg_training_params.yml kore/configs/options/halo_seg_train_ben_params.yml \
    --data.train_set.csv master_annotations_dedup.csv \
    --group driveable_terrain_model \
    --trainer.precision "16-mixed" \
    --optimizer.lr 4e-3 \
    --optimizer.weight_decay 1e-3 \
    --batch_size 72 \
    --model.bn_momentum 1e-1 \
    --ckpt_path /mnt/sandbox1/ben.cline/output/bc_sandbox_2023/cls_dust_light_as_sky_512_640_rgb_no_human_augs_2/bc_sandbox_2023_val_bestmodel.pth \
    --trainer.sync_batchnorm True \
    --run-id ${EXP}_ben_compare
# srun --kill-on-bad-exit python -m kore.scripts.train_seg \
#     --config_path \$CVML_DIR/kore/configs/options/no_val_set.yml \
#     --data.validation_set_ratio 0.05 \
#     --ckpt_path /data/jupiter/alex.li/models/19803_rev1_base_params.ckpt \
#     --optimizer.lr 1e-3 \
#     --optimizer.weight_decay 3e-3 \
#     --warm_up_steps 2000 \
#     --finetuning.enable true \
#     --batch_size 16 \
#     --augmentation.albumentation_transform_path \$CVML_DIR/kore/configs/data/albumentations/seg_halo.yml \
#     --data.train_set.csv master_annotations.csv \
#     --data.train_set.dataset_path /data2/jupiter/datasets/halo_rgb_stereo_train_v6_1 \
#     --run_id ${EXP}_r2_rgbd \
#     --trainer.enable_early_stopping false \
#     --trainer.precision 32 \
#     --output_dir /mnt/sandbox1/$USER/train_halo/\$RUN_ID

# srun --kill-on-bad-exit python -m kore.scripts.train_seg \
#     --config_path \$CVML_DIR/kore/configs/options/no_val_set.yml \
#     --data.validation_set_ratio 0.05 \
#     --ckpt_path /data/jupiter/alex.li/models/19803_rev1_base_params.ckpt \
#     --optimizer.lr 5e-2 \
#     --optimizer.weight_decay 3e-3 \
#     --warm_up_steps 2000 \
#     --finetuning.enable true \
#     --batch_size 16 \
#     --augmentation.albumentation_transform_path \$CVML_DIR/kore/configs/data/albumentations/seg_halo.yml \
#     --data.train_set.csv master_annotations.csv \
#     --data.train_set.dataset_path /data2/jupiter/datasets/halo_rgb_stereo_train_v6_1 \
#     --run_id $EXP_r2 \
#     --trainer.enable_early_stopping false \
#     --trainer.precision 32 \
#     --output_dir /mnt/sandbox1/$USER/train_halo/\$RUN_ID
