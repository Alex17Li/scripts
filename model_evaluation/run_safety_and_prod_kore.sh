#!/bin/bash
#SBATCH --job-name=test_model
#SBATCH --output=/home/%u/logs/%A_%x
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-gpu=6
#SBATCH --time=10:00:00

# Setup environment variables
source /home/alex.li/.bashrc

cd /home/$USER/git/JupiterCVML

CHECKPOINT_FULL_DIR=/mnt/sandbox1/alex.li/wandb/run-20231103_084550-16377/files
DUST_OUTPUT_PARAMS='{"dust_head_output":false}'
CHECKPOINT=last.ckpt
LABEL_MAP_FILE=$CVML_PATH/europa/base/src/europa/dl/config/label_maps/seven_class_train.csv 
CHECKPOINT_FULL_PATH=${CHECKPOINT_FULL_DIR}/${CHECKPOINT}

echo --------------TESTING MODEL $CHECKPOINT_FULL_PATH---------------------------
set -x

SAFETY_DATASETS=("humans_on_path_test_set_2023_v15_anno" "humans_off_path_test_set_2023_v3_anno")
SAFETY_DATASETS=("humans_on_path_test_set_2023_v15_anno")
for DATASET in ${SAFETY_DATASETS[@]}
do
    echo ----------------------------RUN ON ${DATASET}-----------------------------------
    srun --kill-on-bad-exit python -m kore.scripts.predict_seg \
        --data.test_set.csv_name master_annotations.csv \
        --data.test_set.dataset_path /data/jupiter/datasets/$DATASET \
        --inputs.label.label_map_file $LABEL_MAP_FILE \
        --ckpt_path $CHECKPOINT_FULL_PATH \
        --output_dir ${CHECKPOINT_FULL_DIR}/${DATASET} \
        --states_to_save 'human_false_negative' \
        --batch_size 32
done
echo ----------------------------RUN_SAFETY_COMPLETE-----------------------------------

# PROD_DATASETS=("Jupiter_productivity_test_2023_v1_cleaned" "Jupiter_productivity_test_spring_2023_v2_cleaned" "Jupiter_productivity_airborne_debris_first_night")
PROD_DATASETS=()
for DATASET in ${PROD_DATASETS[@]}
do
    echo ----------------------------RUN ON ${DATASET}-----------------------------------
    srun --kill-on-bad-exit python -m kore.scripts.predict_seg \
        --data.test_set.csv_name master_annotations.csv \
        --data.test_set.dataset_path /data/jupiter/datasets/$DATASET \
        --inputs.label.label_map_file $LABEL_MAP_FILE \
        --ckpt_path $CHECKPOINT_FULL_PATH \
        --output_dir ${CHECKPOINT_FULL_DIR}/${DATASET} \
        --states_to_save 'human_false_negative' \
        --metrics.run-productivity-metrics \
        --batch_size 32
done

echo --------------------------RUN_PRODUCTIVITY_COMPLETE-------------------------------
