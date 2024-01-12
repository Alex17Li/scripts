#!/bin/bash
#SBATCH --job-name=kore_rev1_test
#SBATCH --output=/home/%u/logs/%A_%x
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-gpu=6
#SBATCH --time=10:00:00

# Setup environment variables
source /home/alex.li/.bashrc

cd /home/$USER/git/JupiterCVML

DUST_OUTPUT_PARAMS='{"dust_head_output":true}'
LABEL_MAP_FILE=$CVML_PATH/europa/base/src/europa/dl/config/label_maps/seven_class_train.csv 
CHECKPOINT_FULL_DIR=/mnt/sandbox1/alex.li/models
CHECKPOINT=19563.ckpt
# CHECKPOINT_FULL_DIR=/data/jupiter/models/
# CHECKPOINT=v188_58d_rak_local_fine_tversky11_sum_image_normT_prod5_airdyn_r3a8_s30.pth

CHECKPOINT_FULL_PATH=${CHECKPOINT_FULL_DIR}/${CHECKPOINT}

echo --------------TESTING MODEL $CHECKPOINT_FULL_PATH---------------------------

SAFETY_DATASETS=("humans_on_path_test_set_2023_v15_anno" "humans_off_path_test_set_2023_v3_anno")
# SAFETY_DATASETS=()
for DATASET in ${SAFETY_DATASETS[@]}
do
    echo ----------------------------RUN ON ${DATASET}-----------------------------------
    srun --kill-on-bad-exit python -m kore.scripts.predict_seg \
        --data.test_set.csv_name master_annotations.csv \
        --data.test_set.dataset_path /data/jupiter/datasets/$DATASET \
        --inputs.label.label_map_file $LABEL_MAP_FILE \
        --ckpt_path $CHECKPOINT_FULL_PATH \
        --output_dir ${CHECKPOINT_FULL_DIR}/${DATASET} \
        --model.model_params.structural_reparameterization_on_stem true \
        --metrics.use-depth-threshold \
        --states_to_save 'human_false_negative' \
        --batch_size 32 \
        --predictor.precision '32'
    done

echo ----------------------------RUN_SAFETY_COMPLETE-----------------------------------

PROD_DATASETS=("Jupiter_productivity_test_2023_v1_cleaned" "Jupiter_productivity_test_spring_2023_v2_cleaned" "Jupiter_productivity_airborne_debris_first_night")
# PROD_DATASETS=()
for DATASET in ${PROD_DATASETS[@]}
do
    echo ----------------------------RUN ON ${DATASET}-----------------------------------
    srun --kill-on-bad-exit python -m kore.scripts.predict_seg \
        --data.test_set.csv_name master_annotations.csv \
        --data.test_set.dataset_path /data/jupiter/datasets/$DATASET \
        --inputs.label.label_map_file $LABEL_MAP_FILE \
        --ckpt_path $CHECKPOINT_FULL_PATH \
        --model.model_params.structural_reparameterization_on_stem true \
        --output_dir ${CHECKPOINT_FULL_DIR}/${DATASET} \
        --states_to_save 'human_false_positive' \
        --metrics.run-productivity-metrics \
        --inputs.with_semantic_label false \
        --metrics.use-depth-threshold \
        --batch_size 32 \
        --predictor.precision '32'
done

echo --------------------------RUN_PRODUCTIVITY_COMPLETE-------------------------------

DUST_DATASETS=("dust_test_2022_v4_anno_HEAVY_DUST")
for DATASET in ${DUST_DATASETS[@]}
do

    echo ----------------------------RUN ON ${DATASET} labeled ----------------------------
    srun --kill-on-bad-exit python -m kore.scripts.predict_seg \
        --data.test_set.csv_name master_annotations_labeled.csv \
        --data.test_set.dataset_path /data/jupiter/datasets/${DATASET} \
        --inputs.label.label_map_file $LABEL_MAP_FILE \
        --inputs.label.label_map_file_iq $CVML_PATH/europa/base/src/europa/dl/config/label_maps/binary_dust.csv \
        --model.model_params.structural_reparameterization_on_stem true \
        --ckpt_path  ${CHECKPOINT_FULL_PATH} \
        --output-dir ${CHECKPOINT_FULL_DIR}/${DATASET}_labeled \
        --predictor.precision '32' \
        --batch-size 32;
done

echo --------------------------RUN_DUST_COMPLETE---------------------------------------
