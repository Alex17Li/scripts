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

# CHECKPOINT_FULL_PATH=/mnt/sandbox1/alex.li/train_seg_halo/25691_cnp/checkpoints/last.ckpt
# CHECKPOINT_FULL_PATH=/mnt/sandbox1/alex.li/train_seg_halo/25690_cutmix/checkpoints/last.ckpt
# CHECKPOINT_FULL_PATH=/mnt/sandbox1/alex.li/train_seg_halo/25689_dust/checkpoints/last.ckpt
# CHECKPOINT_FULL_PATH=/mnt/sandbox1/alex.li/train_seg_halo/25688_ta/checkpoints/last.ckpt
CHECKPOINT_FULL_PATH=/mnt/sandbox1/alex.li/train_seg_halo/25683_base/checkpoints/last.ckpt
RUN_ID=$(awk -F/ '{print $(NF-2)}' <<< $CHECKPOINT_FULL_PATH)
# CONFIG_PATH=/home/alex.li/git/scripts/training/halo_7_class_pred.yml
# CONFIG_PATH=/home/alex.li/git/scripts/training/halo_8_class_pred.yml
CONFIG_PATH=/home/alex.li/git/scripts/training/halo_9_class_pred.yml

echo --------------TESTING MODEL $CHECKPOINT_FULL_PATH---------------------------
SAFETY_DATASETS=(
    # halo_humans_on_path_test_v6_2_8_mainline
    # on_path_aft_humans_day_2024_rev2_v2
    halo_humans_on_path_test_v6_2_2_test_dataset
)
declare -A PRODUCTIVITY_DATASETS
PRODUCTIVITY_DATASETS[halo_productivity_combined_removing_mislocalization]=data2


for DATASET_NAME in ${SAFETY_DATASETS[@]}
do

    srun --kill-on-bad-exit python -m kore.scripts.predict_seg \
        --config_path $CONFIG_PATH \
        --data.test_set.csv master_annotations.csv \
        --data.test_set.dataset_path /data2/jupiter/datasets/$DATASET_NAME \
        --ckpt_path $CHECKPOINT_FULL_PATH \
        --output_dir /mnt/sandbox1/alex.li/introspection/$RUN_ID/$DATASET_NAME \
        --states_to_save '' \
        --run-id $RUN_ID \
        --pred-tag 'kore' \
        --metrics.pred_stop_classes_to_consider "Non-driveable" "Trees_Weeds" "Humans" "Vehicles" \
        --metrics.gt_stop_classes_to_consider "Humans"
done

echo ----------------------------RUN_SAFETY_COMPLETE-----------------------------------


for DATASET_NAME in ${!PRODUCTIVITY_DATASETS[@]}
do
    srun --kill-on-bad-exit python -m kore.scripts.predict_seg \
        --config_path $CONFIG_PATH \
        --data.test_set.csv master_annotations.csv \
        --data.test_set.dataset_path /${PRODUCTIVITY_DATASETS[${DATASET_NAME}]}/jupiter/datasets/$DATASET_NAME \
        --ckpt_path $CHECKPOINT_FULL_PATH \
        --output_dir /mnt/sandbox1/alex.li/introspection/$RUN_ID/$DATASET_NAME \
        --states_to_save '' \
        --run-id $RUN_ID \
        --pred-tag 'kore' \
        --inputs.with_semantic_label false \
        --metrics.run_productivity_metrics \
        --metrics.gt_stop_classes_to_consider "Non-driveable" "Trees_Weeds" "Humans" "Vehicles" \
        --metrics.pred_stop_classes_to_consider "Non-driveable" "Trees_Weeds" "Humans" "Vehicles"
done
