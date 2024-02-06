#!/bin/bash
#SBATCH --job-name=r2_prod_eval
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

CHECKPOINT_FULL_DIR=/data/jupiter/alex.li/models
CHECKPOINT=20676.ckpt
RUN_ID=20676_r2_rgb_decay

CHECKPOINT_FULL_PATH=${CHECKPOINT_FULL_DIR}/${CHECKPOINT}

echo --------------TESTING MODEL $CHECKPOINT_FULL_PATH---------------------------
PROD_DATASETS=("Jupiter_20231121_HHH2_1800_1830")
# PROD_DATASETS=()
for DATASET in ${PROD_DATASETS[@]}
do
    srun --kill-on-bad-exit python -m kore.scripts.predict_seg \
        --config_path /home/alex.li/git/scripts/training/halo_7_class_pred.yml \
        --data.test_set.csv master_annotations.csv \
        --data.test_set.dataset_path /data/jupiter/datasets/$DATASET \
        --ckpt_path $CHECKPOINT_FULL_PATH \
        --output_dir /mnt/sandbox1/alex.li/introspection/$RUN_ID/$DATASET \
        --states_to_save 'false_positive' \
        --metrics.run-productivity-metrics \
        --inputs.with_semantic_label false \
        --metrics.use-depth-threshold \
        --run-id $RUN_ID \
        --pred-tag $CHECKPOINT
done
