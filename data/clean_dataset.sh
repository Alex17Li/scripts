#!/bin/bash
#SBATCH --job-name=clean_dataset
#SBATCH --output=/home/%u/logs/%A_%x.txt
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@bluerivertech.com

# TODO: load your environment here. This assumes that you have loaded your environment in .bashrc
source ~/.bashrc

cd /home/${USER}/git/scripts/data

python clean_dataset.py /data/jupiter/datasets/Spring_hitchhiker_random
