#!/bin/bash
#SBATCH --job-name=seg_dust
#SBATCH --output=/home/%u/logs/%A_%x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --time=150:00:00
source /home/$USER/.bashrc
conda activate cvml

cd /home/$USER/git
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_MIN_CHANNELS=32

export COLUMNS=200
EXP=${SLURM_JOB_ID}

# python -m JupiterCVML.dlcore.scripts.train_seg --trainer.logger.name $EXP

python -m JupiterCVML.dlcore.scripts.train_seg \
    --config_path scripts/dlcore_configs/harvest_seg_train.yml \
    --config_path scripts/dlcore_configs/seg_gsam.yml \
    --trainer.logger.version $EXP
