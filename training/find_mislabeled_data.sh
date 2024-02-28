#!/bin/bash
#SBATCH --job-name=find_mislabeled
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

cd /home/$USER/git/JupiterCVML

export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_MIN_CHANNELS=32
export COLUMNS=100

srun --kill-on-bad-exit python -m kore.scripts.seg_find_mislabeled_data \
    --augmentation.albumentation_transform_path \$CVML_DIR/kore/configs/data/albumentations/seg_trivialaugment.yml \
    --data.train_set.csv master_annotations.csv \
    --data.train_set.dataset_path /data2/jupiter/datasets/halo_rgb_stereo_train_v6_2/ \
    --data.validation_set_ratio 0.05 \
    --finetuning.skip_mismatched_layers True \
    --inputs.input_mode RECTIFIED_RGB \
    --run-id find_mislabeled_data_halo_62 \
    --trainer.callbacks.tqdm false \
    --loss_only true \
    --save_triage_images true \
    --triage_loss_thresholds .001 .0025 .005 .01 .025 .5
