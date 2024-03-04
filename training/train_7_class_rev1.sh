#!/bin/bash
#SBATCH --job-name=r1_seg
#SBATCH --output=/mnt/sandbox1/%u/logs/%A_%x
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --time=200:00:00
#SBATCH --mem-per-gpu=60G

source /home/$USER/.bashrc
conda activate cvml

cd /home/$USER/git/JupiterCVML

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

# CONFIG_PATH="scripts/kore_configs/harvest_seg_train.yml scripts/kore_configs/seg_gsam.yml \$CVML_DIR/koreconfigs/options/seg_no_dust_head.yml"

set -x

# --augmentation.albumentation_transform_path \$CVML_DIR/kore/configs/data/albumentations/seg_trivialaugment.yml \

srun --kill-on-bad-exit python -m kore.scripts.train_seg \
    --trainer.precision 16 \
    --trainer.enable_early_stopping false \
    --model.model_params.structural_reparameterization_on_stem true \
    --model.model_params.use_highres_downsampling true \
    --output_dir /mnt/sandbox1/$USER/train_rev1/\$RUN_ID \
    --run-id ${EXP}_rev1_downsampling

#     --data.validation_set.csv v6_2_overlap_with_test_geohash_bag_vat_ids.csv \
#     --data.validation_set.dataset_path /data2/jupiter/datasets/Jupiter_train_v6_2 \
#     --data.validation_set.absolute_csv false
