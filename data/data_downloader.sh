#!/bin/bash
#SBATCH --job-name=download_data
#SBATCH --partition=cpu
#SBATCH --mem-per-cpu=15G
#SBATCH --output=/mnt/sandbox1/%u/logs/%A_%x.txt
#SBATCH --ntasks=1

source /home/${USER}/.bashrc

cd /home/${USER}/git/scripts/data

DATASET_NAME=halo_vehicles_in_dust_collection_march2024

python data_downloader.py $DATASET_NAME -d /data2/jupiter/datasets
