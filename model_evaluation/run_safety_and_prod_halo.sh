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

# CHECKPOINT_FULL_PATH=/mnt/sandbox1/alex.li/train_seg_halo/25890_nextvit_rev2_subset/checkpoints/last.ckpt
# CHECKPOINT_FULL_PATH=/mnt/sandbox1/alex.li/train_seg_halo/25892_nextvit_rev2_subset_scale/checkpoints/last.ckpt
# CHECKPOINT_FULL_PATH=/mnt/sandbox1/alex.li/train_seg_halo/25896_nextvit_rev2_subset_scale/checkpoints/last.ckpt
# CHECKPOINT_FULL_PATH=/mnt/sandbox1/alex.li/train_seg_halo/25916_nextvit_rev2_subset_norm/checkpoints/last.ckpt
CHECKPOINT_FULL_PATH=/mnt/sandbox1/alex.li/train_seg_halo/26002_lite12_baseline_all_6_2/checkpoints/last.ckpt
# CHECKPOINT_FULL_PATH=/mnt/sandbox1/alex.li/train_seg_halo/25833_nextvit_rev2/checkpoints/epoch=49.ckpt
# CHECKPOINT_FULL_PATH=/mnt/sandbox1/alex.li/train_seg_halo/26408_lite12_upgrade_all_6_2/checkpoints/last.ckpt

RUN_ID=$(awk -F/ '{print $(NF-2)}' <<< $CHECKPOINT_FULL_PATH)
# CONFIG_PATH=/home/alex.li/git/scripts/training/halo_7_class_pred.yml
# CONFIG_PATH=/home/alex.li/git/scripts/training/halo_8_class_pred.yml
CONFIG_PATH=/home/alex.li/git/scripts/training/halo_9_class_pred.yml
# CONFIG_PATH=/home/alex.li/git/scripts/training/halo_nextvit_pred.yml

echo --------------TESTING MODEL $CHECKPOINT_FULL_PATH---------------------------
SAFETY_DATASETS=(
    halo_humans_on_path_test_v6_2_8_mainline
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
        --output_dir /mnt/sandbox1/alex.li/halo_eval/$RUN_ID/$DATASET_NAME \
        --states_to_save 'false_negative' \
        --run-id $RUN_ID \
        --pred-tag 'kore' \
        --metrics.pred_stop_classes_to_consider "Non-driveable" "Trees_Weeds" "Humans" "Vehicles" \
        --metrics.gt_stop_classes_to_consider "Humans"
done


echo ----------------------------RUN_SAFETY_COMPLETE-----------------------------------


# for DATASET_NAME in ${!PRODUCTIVITY_DATASETS[@]}
# do
#     srun --kill-on-bad-exit python -m kore.scripts.predict_seg \
#         --model.dust.dust_seg_output True \
#         --model.model_params.structural_reparameterization_on_stem True \
#         --inputs.label.label_map_file /home/alex.li/git/scripts/training/label_map_eight_class_birds_as_birds_segdusthead.csv \
#         --inputs.label.label_map_file_iq \$EUROPA_DIR/dl/config/label_maps/binary_dust.csv \
#         --config_path $CONFIG_PATH \
#         --data.test_set.csv master_annotations.csv \
#         --data.test_set.dataset_path /${PRODUCTIVITY_DATASETS[${DATASET_NAME}]}/jupiter/datasets/$DATASET_NAME \
#         --ckpt_path $CHECKPOINT_FULL_PATH \
#         --output_dir /mnt/sandbox1/alex.li/halo_eval/$RUN_ID/$DATASET_NAME \
#         --states_to_save 'false_positive' \
#         --run-id $RUN_ID \
#         --pred-tag 'kore' \
#         --inputs.with_semantic_label false \
#         --metrics.run_productivity_metrics \
#         --metrics.gt_stop_classes_to_consider "Non-driveable" "Trees_Weeds" "Humans" "Vehicles" \
#         --metrics.pred_stop_classes_to_consider "Non-driveable" "Trees_Weeds" "Humans" "Vehicles"
# done
