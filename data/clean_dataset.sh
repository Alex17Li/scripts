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
# python clean_dataset.py $DATASET_PATH/rev1_data_stratified 64cf05781dfbe26adf153573_master_annotations.csv

python clean_dataset.py /data2/jupiter/datasets/20231017_halo_rgb_labeled_excluded_bad_iq

python clean_dataset.py /data/jupiter/datasets/iq_2023_v5_anno
