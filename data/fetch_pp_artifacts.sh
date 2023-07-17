#!/bin/bash
#SBATCH --job-name=fetch_pp_artifacts
#SBATCH --output=/home/%u/logs/%A_%x.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@bluerivertech.com

source ~/.bashrc

cd /home/${USER}/git/JupiterCVML/europa/base/src/europa/dl/dataset

DATASET_PATH=/data/jupiter/datasets

python fetch_pp_artifacts.py \
--output-path ${DATASET_PATH}/spring_dust_data_test \
--master-csv-s3-uri s3://blueriver-jupiter-data/pack_perception/ml/3ea359f0c3fd28d093bf41b44a016d15_d76b3d3f580bfc71d0f9e6ad8d992f2f/64a87abee30a2c394883bc62_master_annotations.csv

