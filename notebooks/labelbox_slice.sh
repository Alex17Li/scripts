#!/bin/bash
#SBATCH --job-name=labelbox_slice
#SBATCH --output=/home/%u/logs/%A_%x.txt
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@bluerivertech.com

source ~/.bashrc

cd /home/${USER}/git/scripts/notebooks
# Takes about 1 minute per 100k images
python labelbox_slice.py --slice_id clri1vxe50lob073e5n9s8rdb \
    --output_path /data/jupiter/alex.li/puddle_slice.parquet
python labelbox_slice.py --slice_id clri4mgxs0c7s071k2erp9yt3 \
    --output_path /data/jupiter/alex.li/dust_slice.parquet