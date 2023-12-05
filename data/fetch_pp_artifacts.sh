#!/bin/bash
#SBATCH --job-name=fetch_pp_artifacts
#SBATCH --output=/home/%u/logs/%A_%x.txt
#SBATCH --partition=cpu
#SBATCH --ntasks=1

source ~/.bashrc

cd /home/${USER}/git/JupiterCVML/europa/base/src/europa/dl/dataset

python fetch_pp_artifacts.py \
--output-path ${DATASET_PATH}/halo_all_cam_labeled_dataset \
--master-csv-s3-uri  s3://blueriver-jupiter-data/pack_perception/ml/1b87a26cceccec5e904b772ae35c33e3_71573a7e6642901f3983f4b0d588b0c7/6553f27dfbfc5c128ad6c27e_master_annotations.csv

python fetch_pp_artifacts.py \
--output-path /data/jupiter/datasets/bad_iq_halo_labelbox_plus_exposure \
--master-csv-s3-uri s3://blueriver-jupiter-data/pack_perception/ml/edf33ddad500682ad81c4c2aec4bdde6_baa7754f69c9985b03cb04a52f5b5bcd/654a5bb2e89875bddc714dd2_master_annotations.csv
