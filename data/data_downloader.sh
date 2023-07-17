#!/bin/bash
#SBATCH --job-name=download_data
#SBATCH --partition=cpu
#SBATCH --output=/home/%u/logs/%A_%x.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1

source /home/${USER}/.bashrc

cd /home/${USER}/git/scripts/data

DATASET_NAME=hhh_field_data_stratified

python data_downloader.py $DATASET_NAME -d $DATASET_PATH

python clean_dataset.py $DATASET_PATH/$DATASET_NAME
