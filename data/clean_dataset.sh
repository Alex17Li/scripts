#!/bin/bash
#SBATCH --job-name=clean_dataset
#SBATCH --output=/home/%u/logs/%A_%x.txt
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@bluerivertech.com

source ~/.bashrc

cd /home/${USER}/git/scripts/data

python clean_dataset.py $DATASET_PATH/all_jupiter_data_stratified cleaned_annotations.csv