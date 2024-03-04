#!/bin/bash
#SBATCH --job-name=r2_eval_seg
#SBATCH --output=/mnt/sandbox1/%u/logs/%A_%x
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-gpu=8
#SBATCH --time=10:00:00
#SBATCH --mem-per-gpu=40G

# Setup environment variables
source /home/alex.li/.bashrc

cd /home/$USER/git/JupiterCVML

CHECKPOINT_FULL_DIR=/data/jupiter/alex.li/models
CHECKPOINT=22976_ben_compare.ckpt
RUN_ID=22976_ben_compare
# CHECKPOINT_FULL_DIR=/mnt/sandbox1/ben.cline/output/bc_sandbox_2023/ds_v6_1_4x_human
# CHECKPOINT=bc_sandbox_2023_val_bestmodel.pth
# RUN_ID=ben_ds_v6_1_4x_human
# CONFIG_PATH=/home/alex.li/git/scripts/training/halo_7_class_pred.yml
CONFIG_PATH=/home/alex.li/git/scripts/training/halo_8_class_pred.yml
CHECKPOINT_FULL_PATH=${CHECKPOINT_FULL_DIR}/${CHECKPOINT}

echo --------------TESTING MODEL $CHECKPOINT_FULL_PATH---------------------------
SAFETY_DATASETS=(
    halo_humans_on_path_test_v6_2_2_test_dataset
)
declare -A PRODUCITIVY_DATASETS
# PRODUCITIVY_DATASETS[halo_rgb_stereo_test_v6_2]=data2
PRODUCITIVY_DATASETS[halo_productivity_combined]=data

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
        --pred-tag $CHECKPOINT \
        --metrics.pred_stop_classes_to_consider "Non-driveable" "Trees_Weeds" "Humans" "Vehicles" \
        --metrics.gt_stop_classes_to_consider "Humans"

done

# echo ----------------------------RUN_SAFETY_COMPLETE-----------------------------------

for DATASET_NAME in ${!PRODUCITIVY_DATASETS[@]}
do
    srun --kill-on-bad-exit python -m kore.scripts.predict_seg \
        --config_path $CONFIG_PATH \
        --data.test_set.csv master_annotations.csv \
        --data.test_set.dataset_path /${PRODUCITIVY_DATASETS[${DATASET_NAME}]}/jupiter/datasets/$DATASET_NAME \
        --ckpt_path $CHECKPOINT_FULL_PATH \
        --output_dir /mnt/sandbox1/alex.li/introspection/$RUN_ID/$DATASET_NAME \
        --states_to_save '' \
        --run-id $RUN_ID \
        --pred-tag $CHECKPOINT \
        --inputs.with_semantic_label false \
        --metrics.run_productivity_metrics \
        --metrics.gt_stop_classes_to_consider "Non-driveable" "Trees_Weeds" "Humans" "Vehicles" \
        --metrics.pred_stop_classes_to_consider "Non-driveable" "Trees_Weeds" "Humans" "Vehicles"
done
