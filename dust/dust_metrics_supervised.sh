#!/bin/bash
#SBATCH --job-name=sup_dust_analysis
#SBATCH --output=/mnt/sandbox1/%u/logs/%A_%x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=30G
#SBATCH --time=8:00:00

source /home/$USER/.bashrc
conda activate cvml

JCVML_PATH=/home/$USER/git/JupiterCVML/europa/base/src/europa
cd $JCVML_PATH
echo ----CURRENT GIT BRANCH AND DIFF----
git branch --show-current
git diff
echo --------------TESTING MODEL $CHECKPOINT_FULL_PATH---------------------------

CHECKPOINT_FULL_PATH=/data/jupiter/li.yu/exps/driveable_terrain_model/v188_58d_rak_local_fine_tversky11_sum_image_normT_prod5_airdyn_r3a8_s30/driveable_terrain_model_val_bestmodel.pth
DATASET=humans_on_path_test_set_2023_v15_anno
echo ----------------------------RUN ON ${DATASET}-----------------------------------
python dl/scripts/predictor_pl.py \
    --csv-path /data/jupiter/datasets/${DATASET}/master_annotations.csv \
    --data-dir /data/jupiter/datasets/${DATASET} \
    --dataset ${DATASET}/ \
    --label-map-file /home/li.yu/code/JupiterCVML/europa/base/src/europa/dl/config/label_maps/seven_class_train.csv \
    --restore-from ${CHECKPOINT_FULL_PATH} \
    --output-dir ${OUTPUT_PATH}/${DATASET}/7class_prod/results \
    --model brtresnetpyramid_lite12 \
    --normalization-params '{"policy": "tonemap", "alpha": 0.25, "beta": 0.9, "gamma": 0.9, "eps": 1e-6}' \
    --dust-output-params '{"dust_class_ratio": true}' \
    --states-to-save '' \
    --use-depth-threshold \
    --input-dims 4 \
    --batch-size 20 \
    --num-workers 4;
DATASET=vehicles_on_path_test_set_2023_v5_anno
echo ----------------------------RUN ON ${DATASET} li run-----------------------------------
python dl/scripts/predictor_pl.py \
    --csv-path /data/jupiter/datasets/${DATASET}/master_annotations.csv \
    --data-dir /data/jupiter/datasets/${DATASET} \
    --dataset ${DATASET}/ \
    --label-map-file /home/li.yu/code/JupiterCVML/europa/base/src/europa/dl/config/label_maps/seven_class_train.csv \
    --restore-from ${CHECKPOINT_FULL_PATH} \
    --output-dir ${OUTPUT_PATH}/${DATASET}/7class_prod/results \
    --model brtresnetpyramid_lite12 \
    --normalization-params '{"policy": "tonemap", "alpha": 0.25, "beta": 0.9, "gamma": 0.9, "eps": 1e-6}' \
    --dust-output-params '{"dust_class_ratio": true}' \
    --states-to-save '' \
    --use-depth-threshold \
    --input-dims 4 \
    --batch-size 20 \
    --num-workers 4;