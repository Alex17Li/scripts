#!/bin/bash
#SBATCH --job-name=test_model
#SBATCH --output=/home/%u/logs/%A_%x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=20G
#SBATCH --time=10:00:00

# Setup environment variables
source /home/alex.li/.bashrc

cd /home/$USER/git/JupiterCVML/europa/base/src/europa

CHECKPOINT=dust_val_bestmodel.pth
# CHECKPOINT_FULL_DIR=/mnt/sandbox1/alex.li/dust/13881_hs15e-1
# CHECKPOINT_FULL_DIR=/mnt/sandbox1/alex.li/dust/14110
CHECKPOINT_FULL_DIR=/mnt/sandbox1/alex.li/dust/13881_hs15e-1/

CHECKPOINT_FULL_PATH=${CHECKPOINT_FULL_DIR}/${CHECKPOINT}

if [ ! -d $CHECKPOINT_FULL_PATh ]; then
    echo checkpoint $CHECKPOINT_FULL_PATH does not exist, please download it.
    echo aws s3 cp s3://mesa-states/prod/jupiter/model_training/{experiment_name}/${CHECKPOINT} ${CHECKPOINT_FULL_DIR}/
    exit 1
fi

echo --------------TESTING MODEL $CHECKPOINT_FULL_PATH---------------------------

DUST_DATASETS=("dust_test_2022_v4_anno_HEAVY_DUST")
for DATASET in ${DUST_DATASETS[@]}
do

echo ----------------------------RUN ON ${DATASET} labeled -----------------------------------
python dl/scripts/predictor.py \
    --csv-path /data/jupiter/datasets/${DATASET}/master_annotations_labeled.csv \
    --data-dir /data/jupiter/datasets/${DATASET} \
    --label-map-file $CVML_PATH/europa/base/src/europa/dl/config/label_maps/seven_class_train.csv \
    --label-map-file-iq $CVML_PATH/europa/base/src/europa/dl/config/label_maps/binary_dust.csv \
    --restore-from  ${CHECKPOINT_FULL_PATH} \
    --output-dir ${CHECKPOINT_FULL_DIR}/${DATASET}_labeled \
    --model brtresnetpyramid_lite12 \
    --normalization-params '{"policy": "tonemap", "alpha": 0.25, "beta": 0.9, "gamma": 0.9, "eps": 1e-6}' \
    --model-params "{\"activation\": \"relu\"}" \
    --dust-output-params "{\"dust_seg_output\": true}" \
    --states-to-save '' \
    --use-depth-threshold \
    --batch-size 20 \
    --tqdm

echo ----------------------------RUN ON ${DATASET} skipped -----------------------------------

python dl/scripts/predictor.py \
    --csv-path /data/jupiter/datasets/${DATASET}/master_annotations_skipped.csv \
    --data-dir /data/jupiter/datasets/${DATASET} \
    --label-map-file $CVML_PATH/europa/base/src/europa/dl/config/label_maps/seven_class_train.csv \
    --label-map-file-iq $CVML_PATH/europa/base/src/europa/dl/config/label_maps/binary_dust.csv \
    --restore-from  ${CHECKPOINT_FULL_PATH} \
    --output-dir ${CHECKPOINT_FULL_DIR}/${DATASET}_skipped \
    --model brtresnetpyramid_lite12 \
    --normalization-params '{"policy": "tonemap", "alpha": 0.25, "beta": 0.9, "gamma": 0.9, "eps": 1e-6}' \
    --model-params "{\"activation\": \"relu\"}" \
    --dust-output-params "{\"dust_seg_output\": true}" \
    --states-to-save '' \
    --use-depth-threshold \
    --batch-size 20 \
    --run-productivity-metrics \
    --tqdm

done

# echo ----------------------------RUN_DUST_COMPLETE-----------------------------------