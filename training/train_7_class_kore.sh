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

EXP=${SLURM_JOB_ID}

python -m JupiterCVML.kore.scripts.train_seg \
    --config_path scripts/kore_configs/harvest_seg_train.yml \
    --trainer.logger.version $EXP

#scripts/kore_configs/seg_gsam.yml \

