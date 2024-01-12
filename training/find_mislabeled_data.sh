#!/bin/bash
#SBATCH --job-name=find_mislabeled
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

# --ckpt_path /data/jupiter/models/cls_dust_light_as_sky_512_640_rgb_no_human_augs_2.pth \
srun --kill-on-bad-exit python -m kore.scripts.seg_find_mislabeled_data \
    --warm_up_steps 1000 \
    --run-id find_mislabled_data \
    --create_new_output_dir_if_nonempty false \
    --trainer.callbacks.tqdm false


srun --kill-on-bad-exit python -m kore.scripts.seg_find_mislabeled_data \
    --augmentation.albumentation_transform_path \$CVML_DIR/kore/configs/data/albumentations/seg_halo.yml \
    --data.train_set.csv_name master_annotations.csv \
    --data.train_set.dataset_path /data2/jupiter/datasets/halo_rgb_stereo_train_v6_1/ \
    --data.validation_set_ratio 0.05 \
    --finetuning.skip_mismatched_layers True \
    --inputs.input_mode RECTIFIED_RGB \
    --warm_up_steps 1000 \
    --run-id find_mislabeled_data_halo_2 \
    --create_new_output_dir_if_nonempty false \
    --trainer.callbacks.tqdm false
