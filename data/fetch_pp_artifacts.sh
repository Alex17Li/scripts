#!/bin/bash
#SBATCH --job-name=fetch_pp_artifacts
#SBATCH --output=/home/%u/logs/%A_%x.txt
#SBATCH --partition=cpu
#SBATCH --ntasks=1

source ~/.bashrc

cd /home/${USER}/git/JupiterCVML/europa/base/src/europa/dl/dataset

# python fetch_pp_artifacts.py \
# --output-path ${DATASET_PATH}/rev2_data_stratified \
# --master-csv-s3-uri s3://blueriver-jupiter-data/pack_perception/ml/3ea359f0c3fd28d093bf41b44a016d15_baa7754f69c9985b03cb04a52f5b5bcd/64cae39a0a0438ef306c214d_master_annotations.csv
python fetch_pp_artifacts.py --output-path ${DATASET_PATH}/Jupiter_train_v5_11 --df-path $DATASET_PATH/master_annotations_30k.csv
