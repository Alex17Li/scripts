#!/bin/bash
#SBATCH --job-name=r2_train_nextvit
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

# srun --kill-on-bad-exit python /home/alex.li/git/scripts/training/test.py
srun --kill-on-bad-exit python /home/$USER/git/JupiterCVML/kore/scripts/train_seg.py \
    --config_path \$CVML_DIR/kore/configs/defaults/halo_seg_training_params.yml \$CVML_DIR/kore/configs/options/highres_experiments_training_params.yml \
        /home/alex.li/git/scripts/training/nextvit.yml \
    --batch_size 28 \
    --run-id ${EXP}_nextvit_rev2_subset_norm \
    --loss.weight_norm_coef .01 \
    # --data.train_set.csv master_annotations_dedup_clean_20240206_okaudit.csv \
    # --ckpt_path /mnt/sandbox1/alex.li/train_seg_halo/25674_nextvit_rev1/checkpoints/epoch=27.ckpt \

    # --augmentation.cutmix.apply_p 0.8 \

    # --data.validation_set.dataset_path /data2/jupiter/datasets/Jupiter_train_v6_2 \
    # --data.validation_set.csv \$CVML_DIR/europa/base/src/europa/dl/config/val_ids/Jupiter_train_v6_2_val_ids_geohash_bag_vat.csv \
    # --data.validation_set.absolute_csv true \
    # --data.train_set.dataset_path /data2/jupiter/datasets/Jupiter_train_v6_2 \
    # --data.train_set.csv master_annotations_20231019_clean.csv \

