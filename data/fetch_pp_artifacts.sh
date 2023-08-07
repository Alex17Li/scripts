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
--output-path ${DATASET_PATH}/mannequin_in_dust \
--master-csv-s3-uri s3://blueriver-jupiter-data/pack_perception/ml/5848694c54f5482c462fd149f971b07e_c7dcb27f46b4e853454683774b99bf3a/64cd53dcc27c05743d53bbdb_master_annotations.csv

python fetch_pp_artifacts.py \
--output-path ${DATASET_PATH}/suv_driving_through_rear_dust_anno \
--master-csv-s3-uri s3://blueriver-jupiter-data/pack_perception/ml/5848694c54f5482c462fd149f971b07e_c7dcb27f46b4e853454683774b99bf3a/64cd53a3748e0a51e1a72774_master_annotations.csv
