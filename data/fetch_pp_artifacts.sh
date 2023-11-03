#!/bin/bash
#SBATCH --job-name=fetch_pp_artifacts
#SBATCH --output=/home/%u/logs/%A_%x.txt
#SBATCH --partition=cpu
#SBATCH --ntasks=1

source ~/.bashrc

cd /home/${USER}/git/JupiterCVML/europa/base/src/europa/dl/dataset

# python fetch_pp_artifacts.py \
# --output-path ${DATASET_PATH}/iq_2023_v5_anno \
# --master-csv-s3-uri  s3://blueriver-jupiter-data/pack_perception/ml/48fe80193177bc671b32ffe6443142c9_e9102e029ccca1fdad1f8dbf60030281/64dfcc1de5a41169c7deb205_master_annotations.csv

python fetch_pp_artifacts.py \
--output-path /data2/jupiter/datasets/20231017_halo_rgb_labeled_excluded_bad_iq \
--master-csv-s3-uri s3://blueriver-jupiter-data/pack_perception/ml/edf33ddad500682ad81c4c2aec4bdde6_baa7754f69c9985b03cb04a52f5b5bcd/653a7a0a3c2d8ab221f6d915_master_annotations.csv