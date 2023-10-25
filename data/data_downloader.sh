#!/bin/bash
#SBATCH --job-name=download_data
#SBATCH --partition=cpu
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/home/%u/logs/%A_%x.txt
#SBATCH --ntasks=1

source /home/${USER}/.bashrc

cd /home/${USER}/git/scripts/data

DATASET_NAME=20231017_halo_rgb_labeled_excluded_bad_iq

python data_downloader.py $DATASET_NAME -d /data2/jupiter/datasets

DATASET_NAME=iq_2023_v5_anno

python data_downloader.py $DATASET_NAME -d /data/jupiter/datasets
