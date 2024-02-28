#!/bin/bash
#SBATCH --job-name=unsup_dust_analysis
#SBATCH --output=/mnt/sandbox1/%u/logs/%A_%x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=30G
#SBATCH --time=150:00:00

source /home/$USER/.bashrc
conda activate cvml

# OUTPUT_PATH=/data/jupiter/$USER/results
JCVML_PATH=/home/$USER/git/JupiterCVML/europa/base/src/europa
cd $JCVML_PATH

DATASET=vehicles_driving_in_dust
ANNOTATIONS_PATH=64dfb36ebe1e14d37b7287d8_master_annotations.csv

# DATASET=rev2_data_stratified
# ANNOTATIONS_PATH=64cae39a0a0438ef306c214d_master_annotations.csv

# python /home/${USER}/git/scripts/data/clean_dataset.py $DATASET_PATH/$DATASET annotations.csv
# python /home/$USER/git/scripts/data/fake_master.py ${DATASET_PATH}/$DATASET
echo "Starting"

python dl/scripts/predictor.py \
    --csv-path ${DATASET_PATH}/${DATASET}/$ANNOTATIONS_PATH \
    --data-dir ${DATASET_PATH}/${DATASET} \
    --label-map-file ${JCVML_PATH}/dl/config/label_maps/four_class_train.csv \
    --restore-from /mnt/sandbox1/alex.li/4class_prod.pth \
    --output-dir ${OUTPUT_PATH}/${DATASET}/results_4class \
    --merge-stop-class-confidence 0.35 \
    --batch-size 32 \
    --dust-class-metrics \
    --model brtresnetpyramid_lite12 \
    --input-mode RGBD \
    --tqdm \
    --states-to-save \
    --gpu all;

# --run-productivity-metrics \
