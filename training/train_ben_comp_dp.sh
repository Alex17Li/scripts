#!/bin/bash
#SBATCH --job-name=dp_r2_train_seg
#SBATCH --output=/mnt/sandbox1/%u/logs/%A_%x
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=150:00:00
#SBATCH --mem-per-gpu=60G
source /home/$USER/.bashrc
conda activate cvml

cd /mnt/sandbox1/$USER/

export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_MIN_CHANNELS=32
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CILK_NWORKERS=1
export TBB_MAX_NUM_THREADS=1
export PL_GLOBAL_SEED=304
export COLUMNS=100

EXP=${SLURM_JOB_ID}

set -x
python /home/$USER/git/JupiterCVML/kore/scripts/train_seg.py \
    --config_path /home/$USER/git/JupiterCVML/kore/configs/defaults/halo_seg_training_params.yml /home/alex.li/git/scripts/training/halo_seg_train_ben_params.yml \
    --ckpt_path /mnt/sandbox1/ben.cline/output/bc_sandbox_2023/cls_dust_light_as_sky_512_640_rgb_no_human_augs_2/bc_sandbox_2023_val_bestmodel.pth \
    --run-id ${EXP}_dp \
    --trainer.use_brt_trainer \
    --batch_size 72 \
    --trainer.num_sanity_val_steps 2 \
