#!/bin/bash
#SBATCH --job-name=download_data
#SBATCH --partition=cpu
#SBATCH --output=/home/%u/logs/%A_%x.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1

source /home/${USER}/.bashrc

cd /home/${USER}/git/scripts/data

DATASET_NAME=hhh_field_data_stratified

python data_downloader.py $DATASET_NAME /data/jupiter/datasets

python clean_dataset.py /data/jupiter/datasets/$DATASET_NAME
