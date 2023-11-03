#!/bin/bash
#SBATCH --job-name=test_model
#SBATCH --output=/home/%u/logs/%A_%x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=20G
#SBATCH --time=10:00:00

# Setup environment variables
source /home/alex.li/.bashrc

cd /home/$USER/git/JupiterCVML/europa/base/src/europa

# experiment name
# check if dir and file exists
CHECKPOINT_FULL_DIR=/mnt/sandbox1/alex.li/wandb/wandb/run-20231018_184348-4c26c094/files
# DUST_OUTPUT_PARAMS='{"dust_head_output":false}'
# DUST_OUTPUT_PARAMS='{"dust_seg_output":true}'
# CHECKPOINT_FULL_DIR=/mnt/sandbox1/alex.li/results/dust_51_v188_58d_rak_local_fine_tversky11_sum_image_normT_prod5_airdyn_r3a8_s30/
DUST_OUTPUT_PARAMS='{"dust_head_output":false}'
# CHECKPOINT=dust_val_bestmodel.pth
CHECKPOINT=epoch=99-val_acc=0.000000-val_loss=0.050370.ckpt
LABEL_MAP_FILE=$CVML_PATH/europa/base/src/europa/dl/config/label_maps/seven_class_train.csv 
CHECKPOINT_FULL_PATH=${CHECKPOINT_FULL_DIR}/${CHECKPOINT}
if [ ! -f $CHECKPOINT_FULL_PATH ]; then
    echo checkpoint does not exist, please run the cmd below to download
    echo aws s3 cp s3://mesa-states/prod/jupiter/model_training/{experiment_name}/${CHECKPOINT} ${CHECKPOINT_FULL_DIR}/
    exit 1
fi

echo --------------TESTING MODEL $CHECKPOINT_FULL_PATH---------------------------

SAFETY_DATASETS=("humans_on_path_test_set_2023_v15_anno" "humans_off_path_test_set_2023_v3_anno")
SAFETY_DATASETS=()
for DATASET in ${SAFETY_DATASETS[@]}
do
# --label-map-file-iq $CVML_PATH/europa/base/src/europa/dl/config/label_maps/binary_dust.csv \
echo ----------------------------RUN ON ${DATASET}-----------------------------------
python dl/scripts/predictor_pl.py \
    --csv-path /data/jupiter/datasets/${DATASET}/master_annotations.csv \
    --data-dir /data/jupiter/datasets/${DATASET} \
    --label-map-file $LABEL_MAP_FILE \
    --restore-from  ${CHECKPOINT_FULL_PATH} \
    --dust-output-params ${DUST_OUTPUT_PARAMS} \
    --output-dir ${CHECKPOINT_FULL_DIR}/${DATASET} \
    --model brtresnetpyramid_lite12 \
    --merge-stop-class-confidence 0.35 \
    --normalization-params '{"policy": "tonemap", "alpha": 0.25, "beta": 0.9, "gamma": 0.9, "eps": 1e-6}' \
    --model-params "{\"activation\": \"relu\"}" \
    --use-depth-threshold \
    --batch-size 64 \
    --tqdm \
    --states-to-save 'human_false_negative';
done

echo ----------------------------RUN_SAFETY_COMPLETE-----------------------------------

# main productivity test sets
PROD_DATASETS=("Jupiter_productivity_test_2023_v1_cleaned" "Jupiter_productivity_test_spring_2023_v2_cleaned" "Jupiter_productivity_airborne_debris_first_night")
# PROD_DATASETS=()
for DATASET in ${PROD_DATASETS[@]}
do

echo ----------------------------RUN ON ${DATASET}-----------------------------------
python dl/scripts/predictor_pl.py \
    --csv-path /data/jupiter/datasets/${DATASET}/master_annotations.csv \
    --data-dir /data/jupiter/datasets/${DATASET} \
    --label-map-file $LABEL_MAP_FILE \
    --restore-from ${CHECKPOINT_FULL_PATH} \
    --output-dir ${CHECKPOINT_FULL_DIR}/${DATASET} \
    --model brtresnetpyramid_lite12 \
    --merge-stop-class-confidence 0.35 \
    --normalization-params '{"policy": "tonemap", "alpha": 0.25, "beta": 0.9, "gamma": 0.9, "eps": 1e-6}' \
    --states-to-save 'human_false_positive' \
    --use-depth-threshold \
    --run-productivity-metrics \
    --batch-size 64;
done

echo --------------------------RUN_PRODUCTIVITY_COMPLETE-------------------------------

# DUST_DATASETS=("dust_test_2022_v4_anno_HEAVY_DUST")
# for DATASET in ${DUST_DATASETS[@]}
# do

#     echo ----------------------------RUN ON ${DATASET} labeled ----------------------------
#     python dl/scripts/predictor_pl.py \
#         --csv-path /data/jupiter/datasets/${DATASET}/master_annotations_labeled.csv \
#         --data-dir /data/jupiter/datasets/${DATASET} \
#         --label-map-file $LABEL_MAP_FILE \
#         --label-map-file-iq $CVML_PATH/europa/base/src/europa/dl/config/label_maps/binary_dust.csv \
#         --restore-from  ${CHECKPOINT_FULL_PATH} \
#         --dust-output-params ${DUST_OUTPUT_PARAMS} \
#         --output-dir ${CHECKPOINT_FULL_DIR}/${DATASET}_labeled \
#         --model brtresnetpyramid_lite12 \
#         --normalization-params '{"policy": "tonemap", "alpha": 0.25, "beta": 0.9, "gamma": 0.9, "eps": 1e-6}' \
#         --model-params "{\"activation\": \"relu\"}" \
#         --dust-output-params $DUST_OUTPUT_PARAMS \
#         --use-depth-threshold \
#         --batch-size 64;
# done

# echo --------------------------RUN_DUST_COMPLETE---------------------------------------
