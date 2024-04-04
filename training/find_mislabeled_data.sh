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
    --config_path  \$CVML_DIR/kore/configs/defaults/halo_seg_training_params.yml \$CVML_DIR/kore/configs/options/highres_experiments_training_params.yml /home/alex.li/git/scripts/training/nextvit.yml /home/alex.li/git/scripts/training/mislabeled.yml \
    --data.train_set.csv master_annotations_cleaned_20240329.csv \
    --data.train_set.dataset_path /data2/jupiter/datasets/halo_rgb_stereo_train_v8_1/ \
    --data.validation_set.dataset_path /data2/jupiter/datasets/halo_rgb_stereo_train_v8_1/ \
    --data.validation_set.csv halo_rgb_stereo_train_v6_2_val_by_geohash_6_for_50k_subset_okaudit.csv \
    --data.validation_set.absolute_csv false \
    --data.validation_set_ratio 0.00 \
    --finetuning.skip_mismatched_layers True \
    --inputs.input_mode RECTIFIED_RGB \
    --run-id find_mislabeled_data_halo_81 \
    --trainer.callbacks.tqdm false \
    --loss_only true \
    --save_triage_images true \
    --triage_loss_thresholds .2 .3 .4 .5 .6 \
    --batch_size 28 \
    --loss.weight_norm_coef .001 \
    --trainer.max_epochs 15
