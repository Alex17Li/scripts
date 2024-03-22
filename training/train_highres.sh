#!/bin/bash
#SBATCH --job-name=r2_train_seg
#SBATCH --output=/mnt/sandbox1/%u/logs/%A_%x
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --time=150:00:00
#SBATCH --exclude=stc01sppamxnl004
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
# srun --kill-on-bad-exit python /home/$USER/git/JupiterCVML/kore/scripts/train_seg.py \
#     --config_path \$CVML_DIR/kore/configs/defaults/halo_seg_training_params.yml \$CVML_DIR/kore/configs/options/highres_experiments_training_params.yml \
#     --batch_size 18 \
#     --loss.msl_weight 8 \
#     --run-id ${EXP}_highres

# With dust
# srun --kill-on-bad-exit python /home/$USER/git/JupiterCVML/kore/scripts/train_seg.py \
#     --config_path \$CVML_DIR/kore/configs/defaults/halo_seg_training_params.yml \$CVML_DIR/kore/configs/options/highres_experiments_training_params.yml /home/alex.li/git/scripts/training/dust.yml \
#     --batch_size 18 \
#     --run-id ${EXP}_highres_segdust

DSET_PATH=/data2/jupiter/datasets/halo_rgb_stereo_train_v6_2/
srun --kill-on-bad-exit python /home/$USER/git/JupiterCVML/kore/scripts/train_seg.py \
    --config_path \$CVML_DIR/kore/configs/defaults/halo_seg_training_params.yml \$CVML_DIR/kore/configs/options/highres_experiments_training_params.yml \
    --batch_size 18 \
    --run-id ${EXP}_cutmix_half \
    --augmentation.cutmix.apply_p 0.5
