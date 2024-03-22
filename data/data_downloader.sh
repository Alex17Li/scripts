#!/bin/bash
# SBATCH --job-name=download_data
# SBATCH --partition=cpu
# SBATCH --mem=20G
# SBATCH --cpus-per-task=1
# SBATCH --output=/mnt/sandbox1/%u/logs/%A_%x.txt
# SBATCH --ntasks=1
# SBATCH --time=10:00:00

##SBATCH --partition=gpu
##SBATCH --gres=gpu:1
##SBATCH --mem-per-gpu=120G
source /home/${USER}/.bashrc

cd /home/${USER}/git/scripts/data

# DATASET_NAME=on_path_aft_humans_night_2024_rev2_v2

# python data_downloader.py $DATASET_NAME -d /data2/jupiter/datasets

DATASET_NAME=on_path_aft_humans_day_2024_rev2_v2

python data_downloader.py $DATASET_NAME -d /data2/jupiter/datasets
