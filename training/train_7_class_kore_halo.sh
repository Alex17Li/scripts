#!/bin/bash
#SBATCH --job-name=seg_dust
#SBATCH --output=/home/%u/logs/%A_%x
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --time=150:00:00

source /home/$USER/.bashrc
conda activate cvml

cd /home/$USER/git

export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_MIN_CHANNELS=32
export COLUMNS=100

EXP=${SLURM_JOB_ID}

set -x

CONFIG_PATH="scripts/kore_configs/harvest_seg_train.yml"
srun --kill-on-bad-exit python -m JupiterCVML.kore.scripts.train_seg \
    --config_path \$CVML_DIR/kore/configs/defaults/halo_seg_training_params.yml \
    --trainer.logger.version $EXP
