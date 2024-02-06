#!/bin/bash
#SBATCH --job-name=r2_eval_seg
#SBATCH --output=/home/%u/logs/%A_%x
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-gpu=6
#SBATCH --time=10:00:00
#SBATCH --mem-per-gpu=40G

# Setup environment variables
source /home/alex.li/.bashrc

cd /home/$USER/git/JupiterCVML

# CHECKPOINT_FULL_DIR=/mnt/sandbox1/alex.li/train_halo/20676_r2_rgb_bigdecay_biglr/checkpoints
# CHECKPOINT=last.ckpt
CHECKPOINT_FULL_DIR=/data/jupiter/alex.li/models
CHECKPOINT=20676.ckpt
RUN_ID=20676_r2_rgb_decay

CHECKPOINT_FULL_PATH=${CHECKPOINT_FULL_DIR}/${CHECKPOINT}

echo --------------TESTING MODEL $CHECKPOINT_FULL_PATH---------------------------

srun --kill-on-bad-exit python -m kore.scripts.predict_seg \
    --config_path /home/alex.li/git/scripts/training/halo_7_class_pred.yml \
    --data.test_set.csv master_annotations.csv \
    --data.test_set.dataset_path /data2/jupiter/datasets/halo_rgb_stereo_test_v6_1_unofficial_cleaned \
    --ckpt_path $CHECKPOINT_FULL_PATH \
    --output_dir /mnt/sandbox1/alex.li/introspection/$RUN_ID/halo_rgb_stereo_test_v6_1_unofficial_cleaned \
    --states_to_save '' \
    --run-id $RUN_ID \
    --pred-tag $CHECKPOINT

    # --metrics.gt_stop_classes_to_consider 'Vehicles' 'Humans' \
#    --states_to_save 'false_negative'

echo ----------------------------RUN_SAFETY_COMPLETE-----------------------------------

