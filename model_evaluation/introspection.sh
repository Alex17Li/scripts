#!/bin/bash
#SBATCH --job-name=r2_eval_seg
#SBATCH --output=/mnt/sandbox1/%u/logs/%A_%x
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-gpu=8
#SBATCH --time=10:00:00
#SBATCH --mem-per-gpu=60G

# Setup environment variables
source /home/alex.li/.bashrc

cd /home/$USER/git/JupiterCVML

CHECKPOINT_FULL_PATH=/mnt/sandbox1/alex.li/train_seg_halo/25833_nextvit_rev2/checkpoints/last.ckpt
RUN_ID=$(awk -F/ '{print $(NF-2)}' <<< $CHECKPOINT_FULL_PATH)
# CONFIG_PATH=/home/alex.li/git/scripts/training/halo_7_class_pred.yml
# CONFIG_PATH=/home/alex.li/git/scripts/training/halo_8_class_pred.yml
# CONFIG_PATH=/home/alex.li/git/scripts/training/halo_9_class_pred.yml
CONFIG_PATH=/home/alex.li/git/scripts/training/halo_nextvit_pred.yml

echo --------------TESTING MODEL $CHECKPOINT_FULL_PATH---------------------------
SAFETY_DATASETS=(
    # off_path_aft_humans_day_2024_rev2_v2
    # off_field_humans_day_2024_rev2_v2
    # off_path_aft_humans_night_2024_rev2_v2
    # on_path_forward_humans_night_2024_rev2_v5
    # on_path_aft_humans_night_2024_rev2_v4
    # on_path_aft_humans_day_2024_rev2_v4
    # on_path_forward_humans_day_2024_rev2_v4
    on_path_forward_humans_night_2024_rev2_v7
    on_path_forward_humans_day_2024_rev2_v6
    on_path_aft_humans_night_2024_rev2_v6
)
# on_path_forward_humans_night_2024_rev2_v3
# on_path_aft_humans_night_2024_rev2_v2
# on_path_aft_humans_day_2024_rev2_v2
# on_path_forward_humans_day_2024_rev2_v2
declare -A PRODUCTIVITY_DATASETS
declare -A PRODUCTIVITY_DATASETS_CSV_FILE
# PRODUCTIVITY_DATASETS[halo_rgb_stereo_test_v6_2_productivity_unoffical_v2]=data
# PRODUCTIVITY_DATASETS[halo_productivity_combined_no_mislocalization_alleyson]=data2


for DATASET_NAME in ${SAFETY_DATASETS[@]}
do

    srun --kill-on-bad-exit python -m kore.scripts.predict_seg \
        --config_path $CONFIG_PATH \
        --data.test_set.csv master_annotations.csv \
        --data.test_set.dataset_path /data2/jupiter/datasets/$DATASET_NAME \
        --ckpt_path $CHECKPOINT_FULL_PATH \
        --output_dir /mnt/sandbox1/alex.li/introspection/$RUN_ID/$DATASET_NAME \
        --states_to_save 'false_positive' 'true_negative' 'false_negative' \
        --run-id $RUN_ID \
        --pred-tag 'kore' \
        --metrics.pred_stop_classes_to_consider "Non-driveable" "Trees_Weeds" "Humans" "Vehicles" \
        --metrics.gt_stop_classes_to_consider "Humans" \
        --metrics.depth_thresholds.T_front_center .5 \
        --metrics.depth_thresholds.T_front_side .5 \
        --metrics.depth_thresholds.T_side_center .5 \
        --metrics.depth_thresholds.T_side_side .5 \
        --metrics.depth_thresholds.T_rear_center .5 \
        --metrics.depth_thresholds.T_rear_side .5 \
        --metrics.depth_thresholds.I_center .5 \
        --metrics.depth_thresholds.I_side .5 \

done

