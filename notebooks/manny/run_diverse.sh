#!/bin/bash
#SBATCH --job-name=diversify_data
#SBATCH --output=/mnt/sandbox1/%u/logs/%A_%x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=120G
#SBATCH --time=8:00:00

source /home/alex.li/.bashrc
conda activate cvml
python /home/alex.li/git/scripts/notebooks/manny/diversify.py labelbox_import_tire_tracks_100k 30000
