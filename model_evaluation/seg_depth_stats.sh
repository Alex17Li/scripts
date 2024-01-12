#!/bin/bash
#SBATCH --job-name=depth_stats
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
srun --kill-on-bad-exit python -m kore.scripts.seg_depth_stats \
    --run-id depth_stats \
    --config_path \$CVML_DIR/kore/configs/options/seg_no_dust_head.yml \$CVML_DIR/kore/configs/defaults/halo_seg_training_params.yml \
    --data.train_set.csv_name master_annotations.csv \
    --data.train_set.dataset_path /data/jupiter/datasets/halo_humans_on_path_test_v6_1/ \
    --finetuning.skip_mismatched_layers True \
    --inputs.input_mode RGBD \
    --ckpt_path \
    --create_new_output_dir_if_nonempty false \
    --inputs.label.label_map_file \$EUROPA_DIR/dl/config/label_maps/eight_class_train_dust_light_as_sky_birds_as_driveable.csv
