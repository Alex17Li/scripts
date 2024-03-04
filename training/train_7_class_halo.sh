#!/bin/bash
#SBATCH --job-name=halo_train_seg
#SBATCH --output=/mnt/sandbox1/%u/logs/%A_%x
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --time=150:00:00
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
RUN_ID=${EXP}_smoothsoftlosstrain_sigmap5
set -x
    # --augmentation.albumentation_transform_path \$CVML_DIR/kore/configs/data/albumentations/seg_trivialaugment_halo.yml \
    # --augmentation.albumentation_transform_path \$CVML_DIR/kore/configs/data/albumentations/seg_halo.yml \

# /home/alex.li/git/scripts/training/dustaug.yml
srun --kill-on-bad-exit python -m kore.scripts.train_seg \
    --optimizer.weight_decay 1e-3 \
    --config_path /home/alex.li/git/scripts/training/halo_7_class_train.yml \
    --augmentation.albumentation_transform_path \$CVML_DIR/kore/configs/data/albumentations/seg.yml \
    --trainer.precision "16-mixed" \
    --optimizer.lr 1e-3 \
    --run_id ${RUN_ID} \
    --output_dir /mnt/sandbox1/$USER/train_halo/\$RUN_ID \
    --trainer.max_epochs 50 \
    --inputs.label.half_res_output True \
    --inputs.label.label_smoothing_sigma 0.5 \
    --inputs.label.label_smoothing_iq_sigma 3 \


#  /home/alex.li/git/scripts/training/dustaug.yml \s
# srun --kill-on-bad-exit python -m kore.scripts.train_seg \
#     --trainer.precision 16-mixed \
#     --optimizer.weight_decay 1e-3 \
#     --config_path /home/alex.li/git/scripts/training/halo_seg_train_ben_params.yml \
#     --trainer.enable_early_stopping false \
#     --output_dir /mnt/sandbox1/$USER/train_halo/\$RUN_ID \
#     --model.model_params.use_highres_downsampling true \
#     --inputs.input-shape 1024 1280 \
#     --run-id ${EXP}_highres \
#     --data.train_set.csv master_annotations_dedup_clean_ocal_20240208_50k_intersection.csv \
#     --data.train_set.dataset_path /data2/jupiter/datasets/halo_rgb_stereo_train_v6_2_full_res/

# --data.train_set.dataset_path /data2/jupiter/datasets/halo_rgb_stereo_train_v6_2 \
# --data.train_set.dataset_path /data2/jupiter/datasets/halo_rgb_stereo_train_v6_2_768 \
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

cp /mnt/sandbox1/$USER/train_halo/$RUN_ID/checkpoints/last.ckpt /data/jupiter/alex.li/models/$RUN_ID.ckpt
