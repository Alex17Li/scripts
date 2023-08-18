#!/bin/bash
#SBATCH --job-name=download_data
#SBATCH --partition=cpu
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/home/%u/logs/%A_%x.txt
#SBATCH --ntasks=1

source /home/${USER}/.bashrc

cd /home/${USER}/git/scripts/data

DATASET_NAME=mannequin_in_dust

python data_downloader.py $DATASET_NAME -d $DATASET_PATH

# python clean_dataset.py $DATASET_PATH/$DATASET_NAME annotations.csv
