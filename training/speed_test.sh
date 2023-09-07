#!/bin/bash
#SBATCH --job-name=dataloader_speed
#SBATCH --output=/home/%u/logs/%A_%x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --time=150:00:00

source /home/$USER/.bashrc
conda activate cvml

cd /home/$USER/git/JupiterCVML/europa/base/src/europa

CVML_PATH=/home/$USER/git/JupiterCVML
EXP=seg_$SLURM_JOB_ID
# EXP=seg_12688
SNAPSHOT_DIR=/mnt/sandbox1/$USER
OUTPUT_DIR=${OUTPUT_PATH}/${EXP}

python -m tests.profiling.speed_test
