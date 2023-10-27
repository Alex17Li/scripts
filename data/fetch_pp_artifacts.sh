#!/bin/bash
#SBATCH --job-name=fetch_pp_artifacts
#SBATCH --output=/home/%u/logs/%A_%x.txt
#SBATCH --partition=cpu
#SBATCH --ntasks=1

source ~/.bashrc

cd /home/${USER}/git/JupiterCVML/europa/base/src/europa/dl/dataset

python fetch_pp_artifacts.py \
--output-path ${DATASET_PATH}/iq_2023_v5_anno \
--master-csv-s3-uri  s3://blueriver-jupiter-data/pack_perception/ml/48fe80193177bc671b32ffe6443142c9_e9102e029ccca1fdad1f8dbf60030281/64dfcc1de5a41169c7deb205_master_annotations.csv